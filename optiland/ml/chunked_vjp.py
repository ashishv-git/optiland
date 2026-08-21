"""Chunked vector-Jacobian product accumulation.

Reverse-mode autograd keeps the whole computational graph alive until the
backward pass runs. For a reduction that sums over many rays -- rendering an
image, accumulating irradiance on a detector -- that graph grows linearly with
the ray count, and its peak size is what caps how many rays a differentiable
design problem can afford.

:func:`chunked_vjp` lifts that cap for reductions that are *additive* over
rays. The total is accumulated once under ``torch.no_grad()``, so the forward
pass builds no graph at all. When the caller's ``loss.backward()`` reaches the
result, each chunk is re-evaluated with gradients enabled, its vector-Jacobian
product is taken against the incoming adjoint, and its graph is freed before
the next chunk is built. Peak graph size is set by the largest chunk rather
than by the total.

This is the adjoint back-propagation of Wang et al., dO section II-D, lifted
out of any particular renderer: nothing here knows what a ray or an image is.

It composes with, rather than replaces, the implicit differentiation used by
the Newton-Raphson solvers (see the Implicit Differentiation developer guide).
Implicit differentiation bounds how *deep* the graph is per ray; this bounds
how *wide* it is across rays.

Ashish Verma, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    import torch
except ImportError:  # pragma: no cover - exercised only without torch
    torch = None

import optiland.backend as be

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


# Subclassing must not blow up at import time when torch is absent: this
# module is imported unconditionally by ``optiland.ml``, and the numpy path
# has to keep working. Mirrors the guard in optiland/ml/wrappers.py.
_AutogradFunction = torch.autograd.Function if torch is not None else object

if torch is not None:
    _once_differentiable = torch.autograd.function.once_differentiable
else:  # pragma: no cover - exercised only without torch

    def _once_differentiable(fn):
        return fn


class _ChunkedVJPFunction(_AutogradFunction):
    """Autograd bridge for :func:`chunked_vjp`.

    Kept private: the supported entry point is :func:`chunked_vjp`, which
    validates its arguments before reaching autograd.

    dO section II-D splits the computation into three stages, and it is worth
    being clear which of them live here:

    1. Accumulate the total with no graph. :meth:`forward`.
    2. Forward and backward through the metric -- a loss, a neural network.
       **Not here.** This is the caller's own ``loss.backward()``; autograd
       reaches this Function only once it is done, and hands the result to
       :meth:`backward` as ``grad_output``.
    3. Backward through the system that produced each chunk.
       :meth:`backward`.

    Stage 2 is what makes the memory saving possible: because the metric has
    already been reduced to a gradient at the total, stage 3 can replay the
    chunks one at a time instead of keeping them all alive.
    """

    @staticmethod
    def forward(ctx, batch_fn, setup_fn, batches, *params):
        """Accumulate the total over chunks without building a graph.

        Args:
            ctx: Autograd context.
            batch_fn: Maps one batch to its contribution to the total.
            setup_fn: Called before each batch, or None.
            batches: The batches to reduce over.
            *params: Tensors to accumulate gradients for.

        Returns:
            torch.Tensor: The summed contributions of every batch.
        """
        total = None
        with torch.no_grad():
            for batch in batches:
                if setup_fn is not None:
                    setup_fn()
                contribution = batch_fn(batch)
                # clone() on the first batch so the result never aliases a
                # tensor the caller still holds a reference to.
                total = contribution.clone() if total is None else total + contribution

        ctx.batch_fn = batch_fn
        ctx.setup_fn = setup_fn
        ctx.batches = batches
        # Tensors go through save_for_backward rather than onto ctx directly:
        # it is what lets autograd track their versions and catch a param
        # mutated between the two passes, which would otherwise produce
        # gradients evaluated at the wrong point.
        ctx.save_for_backward(*params)

        return total

    @staticmethod
    @_once_differentiable
    def backward(ctx, grad_output):
        """Re-evaluate each chunk and accumulate its VJP.

        ``grad_output`` is dL/d(total). Because the total is a plain sum of
        the chunks, d(total)/d(chunk_i) is 1, so the same ``grad_output``
        is the correct incoming adjoint for every chunk -- which is what
        makes chunking valid at all, and why the reduction has to be
        additive.

        Args:
            ctx: Autograd context.
            grad_output (torch.Tensor): Gradient of the loss w.r.t. the total.

        Returns:
            tuple: Gradients matching ``forward``'s signature. The three
            leading ``None`` entries correspond to its non-tensor arguments.
        """
        params = ctx.saved_tensors
        param_grads = [torch.zeros_like(p) for p in params]
        # A param that never receives a VJP from any chunk is a silent-wrong-
        # gradient hazard, so track connectivity and raise rather than
        # returning a confident-looking zero.
        connected = [False] * len(params)

        # enable_grad() is required, not defensive: @once_differentiable runs
        # this method inside no_grad, so without it batch_fn would build no
        # graph and autograd.grad would fail with "element 0 of tensors does
        # not require grad". Removing it does not merely slow things down --
        # it breaks the gradient.
        with torch.enable_grad():
            for batch in ctx.batches:
                if ctx.setup_fn is not None:
                    ctx.setup_fn()

                contribution = ctx.batch_fn(batch)
                if contribution.shape != grad_output.shape:
                    raise ValueError(
                        "batch_fn must return the same shape on every call, "
                        "matching the accumulated total. Got "
                        f"{tuple(contribution.shape)} during the backward "
                        f"pass but {tuple(grad_output.shape)} in the forward "
                        "pass. A chunk is a subset of rays, not a subset of "
                        "the output: every chunk contributes to the whole "
                        "output, so each must return it at full size."
                    )

                vjps = torch.autograd.grad(
                    contribution,
                    params,
                    grad_outputs=grad_output,
                    retain_graph=False,  # this chunk's graph dies here
                    allow_unused=True,
                )

                for i, vjp in enumerate(vjps):
                    if vjp is not None:
                        param_grads[i] += vjp.detach()
                        connected[i] = True

        unused = [i for i, ok in enumerate(connected) if not ok]
        if unused:
            raise RuntimeError(
                f"No gradient reached params at position(s) {unused}. They are "
                "not connected to the graph built by batch_fn, so their "
                "gradients would silently be zero. Check that batch_fn uses "
                "these tensors, and that setup_fn re-injects them into the "
                "system on every call rather than reading them out as plain "
                "values."
            )

        # Break the reference cycle so the batches and closures can be freed.
        ctx.batch_fn = None
        ctx.setup_fn = None
        ctx.batches = None

        return (None, None, None, *param_grads)


def chunked_vjp(
    batch_fn: Callable[[Any], torch.Tensor],
    batches: Sequence[Any],
    params: Sequence[torch.Tensor],
    *,
    setup_fn: Callable[[], None] | None = None,
) -> torch.Tensor:
    """Sum ``batch_fn`` over ``batches`` with graph size independent of batch count.

    The result is a normal autograd tensor: call ``.backward()`` on a loss
    derived from it and the gradients land on ``params`` as usual. What
    differs is that the graph for each chunk is built, used and freed one
    chunk at a time, so peak memory tracks the largest chunk rather than the
    total.

    Args:
        batch_fn: Maps one element of ``batches`` to that batch's
            contribution to the total. Must return the **same shape** on
            every call -- a chunk is a subset of rays, not a subset of the
            output, so every chunk contributes to the whole output. Must be
            deterministic: the backward pass re-runs it and the result has to
            reproduce the forward pass exactly.
        batches: The batches to reduce over, for example index slices into a
            pre-generated set of pupil samples. Passing explicit slices,
            rather than per-chunk random seeds, is what makes the result
            invariant to chunk size.
        params: Tensors to accumulate gradients for. Each must be reachable
            from the graph ``batch_fn`` builds.
        setup_fn: Called before every batch in both passes, for systems that
            need ``params`` re-injected before tracing. Must be idempotent.

    Returns:
        torch.Tensor: The sum of every batch's contribution.

    Raises:
        RuntimeError: If torch is unavailable, if the active backend is not
            torch, or if any entry of ``params`` is unreachable from the
            graph.
        ValueError: If ``batches`` is empty, or if ``batch_fn`` returns
            inconsistent shapes.

    Note:
        Valid only for reductions that are genuinely **additive** over
        batches. It is wrong for ``max``, and for any quantity normalised by
        something derived from all rays at once, such as a centroid
        reference; in those cases the total is not the sum of the chunks and
        the incoming adjoint does not apply unchanged to each of them. This
        is not detected at runtime.

    Example:
        >>> total = chunked_vjp(
        ...     render_chunk,
        ...     [slice(i, i + 10_000) for i in range(0, 1_000_000, 10_000)],
        ...     params=[radius],
        ... )
        >>> loss = criterion(total, target)
        >>> loss.backward()
    """
    if torch is None:
        raise RuntimeError(
            "chunked_vjp requires the 'torch' package. Install PyTorch to "
            "use this function."
        )
    if be.get_backend() != "torch":
        raise RuntimeError(
            "chunked_vjp requires the 'torch' backend, but the active "
            f"backend is '{be.get_backend()}'. This feature accumulates "
            "gradients through PyTorch autograd and has no numpy equivalent; "
            "call optiland.backend.set_backend('torch') first."
        )

    # Copied into a list because both passes iterate it. A generator would be
    # exhausted by the forward pass and yield nothing in the backward one,
    # silently producing zero gradients rather than an error.
    batches = list(batches)
    if not batches:
        raise ValueError("batches is empty; nothing to reduce over.")

    params = tuple(params)
    if not params:
        raise ValueError("params is empty; there is nothing to accumulate.")

    no_grad = [i for i, p in enumerate(params) if not p.requires_grad]
    if no_grad:
        raise ValueError(
            f"params at position(s) {no_grad} do not require grad, so no "
            "gradient can be accumulated for them. Call requires_grad_(True) "
            "on them, or leave them out of params."
        )

    return _ChunkedVJPFunction.apply(batch_fn, setup_fn, batches, *params)
