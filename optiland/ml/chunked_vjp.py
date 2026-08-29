"""Chunked vector-Jacobian product accumulation.

Differentiating a ray trace requires memory proportional to the number of
rays, because reverse-mode autograd retains the whole graph until the
backward pass runs. Many optical quantities are *reductions* over rays -- a
rendered image, an irradiance map -- each accumulating one contribution per
ray into a single total. Differentiating such a reduction over enough rays
produces an autograd graph that exceeds device memory.

This module provides :func:`chunked_vjp`, which lifts that memory limit for
reductions that are additive over rays. In exchange, every ray is traced
twice: once to accumulate the total, and again to differentiate it. Any
per-batch setup work is repeated in each pass, so smaller batches lower the
memory ceiling and raise that overhead.

Given ``batch_fn``, which computes the contribution of one *batch* of rays to
the total, the batches to sum over, and the parameters to differentiate with
respect to, :func:`chunked_vjp` returns the total as an ordinary autograd
tensor for which ``.backward()`` and ``.grad`` behave as usual.

Each batch's graph is built and released before the next is built, so at most
one exists at any moment. Peak graph memory is therefore bounded by the
largest batch, independent of the batch count.

A batch is a subset of the rays. Its rays may contribute to any element of
the total tensor, so a batch's contribution matches the shape of the whole
total. For example, when the total is a rendered image, a batch's
contribution is that whole image, dim and sparsely sampled. A tile of the
image, fully illuminated, would be the wrong shape.

The module does not inspect a batch: each is passed to ``batch_fn``
unchanged, and the computation producing its contribution is unconstrained.
It applies to any additive reduction over rays.

Why an ordinary loop is not enough
----------------------------------
Accumulating the batches in an ordinary loop does not lower peak graph
memory. Every batch's graph remains reachable from the running total, so all
of them are still retained when the backward pass runs.

How the computation is split
----------------------------
The computational graph spans from the optical parameters, through the traced
rays, into the total, and on into the merit function applied to it. That
merit function may be a least-squares comparison against a target, or a
neural network computing a perceptual loss. Reverse-mode autograd traverses
the graph in a single pass, holding the ray-tracing portion in memory
throughout.

The graph can instead be cut at the total. Writing ``L`` for the merit
function, the chain rule gives::

    dL/d(params) = dL/d(total) * d(total)/d(params)

The two factors can be evaluated in separate passes. dO calls this the
**separability property** and uses it to split the computation into three
sequential stages (section II-D):

1. **Forward, no autograd.** :func:`chunked_vjp` accumulates the total over
   the batches under ``torch.no_grad()``, so no graph is built.
2. **Forward and backward on the merit function.** The caller applies the
   merit function to that total, then calls ``.backward()`` on the resulting
   scalar loss. Autograd propagates back through the merit function,
   including any network it contains, and arrives at the total with
   ``dL/d(total)``, a tensor of the same shape.
3. **Forward and backward on the reduction.** For one batch at a time,
   :func:`chunked_vjp` re-evaluates ``batch_fn`` with gradients enabled,
   rebuilding the graph from ``params`` to that batch's contribution to the
   total. That graph encodes the Jacobian ``d(contribution)/d(params)``. The
   vector-Jacobian product of ``dL/d(total)`` with it gives the batch's
   share of ``dL/d(params)``, which is accumulated into the parameter
   gradients. The graph is freed before the next batch is built.

Only stages 2 and 3 run a backward pass; stage 1 is a forward computation
with no autograd. Stage 3 does not evaluate the merit function: it needs only
``dL/d(total)``, which stage 2 has already produced. Each batch can be
differentiated on its own.

Preconditions
-------------
Two conditions must hold for the gradients to be correct: the reduction must
be additive over rays, and ``batch_fn`` must be deterministic. A non-additive
reduction cannot be detected, and yields silently incorrect gradients. A
non-deterministic ``batch_fn`` is detected and raises.

.. warning::
   The reduction must be genuinely **additive** over rays, so that the total
   is the plain sum of the batch contributions and stage 3 can supply every
   batch with the same incoming gradient. This does not hold for ``max``, nor
   for a quantity normalised by something derived from all rays, such as a
   centroid reference. **This cannot be detected**, so a non-additive
   reduction yields silently incorrect gradients. Two facts make it
   undetectable from inside: batches are opaque tokens, so no second
   partition can be built to compare against, and the contributions
   themselves carry no signal about which reduction produced them -- with one
   ray per batch, a sum and a maximum return identical values. Verifying
   additivity is the caller's job; ``test_result_is_invariant_to_batch_size``
   in the test suite is a template for doing it.

.. warning::
   ``batch_fn`` must be **deterministic**. Stage 3 re-evaluates it, and the
   result must reproduce stage 1. Anything stochastic in ray generation or
   scattering needs care here, as does a batch that stage 1 consumes, such as
   a one-shot iterator. **This is detected.** Two guards cover it: iterator
   batches are rejected at the entry point, and the stage-3 contributions are
   summed and compared against the stage-1 total, so a mismatch from any
   other cause raises instead of returning a gradient taken through a
   different quantity.

   Stage 1's global RNG state is also restored before stage 3, so a
   ``batch_fn`` sampling through the torch generators -- as optiland's own
   distributions do -- reproduces without the caller arranging anything. That
   does not extend to a generator the caller holds and advances, nor to numpy
   or Python ``random``; those reach the comparison instead.

   Satisfying the condition still beats relying on the check. Counter-based
   sampling keyed by ray index keeps a stochastic ``batch_fn`` reproducible,
   and index slices or arrays cannot be consumed by the first pass.

Relation to implicit differentiation
------------------------------------
This composes with the implicit differentiation used by the Newton-Raphson
solvers (see the Implicit Differentiation developer guide); it does not
replace it. That bounds how *deep* the graph is per ray; this bounds how
*wide* it is across rays.

Reference
---------
Wang, Chen and Heidrich, "dO: A Differentiable Engine for Deep Lens Design of
Computational Imaging Systems", IEEE Transactions on Computational Imaging,
2022. Section II-D introduces the separability property and calls the
three-stage procedure adjoint back-propagation.

Ashish Verma, 2026
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

try:
    import torch
except ImportError:  # pragma: no cover - exercised only without torch
    torch = None

import optiland.backend as be
from optiland.utils import machine_eps

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


# Subclassing must not blow up at import time when torch is absent: this
# module is imported unconditionally by ``optiland.ml``, and the numpy path
# must keep working. Mirrors the guard in optiland/ml/wrappers.py.
_AutogradFunction = torch.autograd.Function if torch is not None else object

if torch is not None:
    _once_differentiable = torch.autograd.function.once_differentiable
else:  # pragma: no cover - exercised only without torch

    def _once_differentiable(fn):
        return fn


# Multiplier on machine epsilon for the stage-1 / stage-3 consistency check.
# Sized to absorb GPU kernel non-determinism -- scatter_add and index_add
# accumulate in non-deterministic order -- without admitting a genuinely
# non-reproducible batch_fn, whose two passes differ by O(1) rather than by
# round-off. Deliberately generous: the tests run on CPU, where two sums of
# the same terms in the same order are bitwise equal, so a value tuned to CPU
# behaviour would pass CI and then fail on a GPU.
_CONSISTENCY_EPS_MULTIPLIER = 32.0


def _consistency_tolerance(total, n_batches: int) -> float:
    """Absolute tolerance for comparing the stage-1 and stage-3 totals.

    Scales with the working dtype, the magnitude of the total, and the number
    of batches, since the comparison is between two sums of that many terms.

    Args:
        total: The total accumulated by stage 1.
        n_batches (int): Number of batches summed into it.

    Returns:
        float: Absolute tolerance for the comparison.
    """
    # Non-finite entries are normal here -- a vignetted ray contributes NaN --
    # so the scale must come from the finite entries alone. Taking max() over
    # the raw tensor would return NaN and make every comparison fail.
    finite = total[torch.isfinite(total)]
    scale = float(finite.abs().max()) if finite.numel() else 1.0
    return (
        _CONSISTENCY_EPS_MULTIPLIER
        * machine_eps(total)
        * max(1.0, scale)
        * max(1, n_batches)
    )


class _ChunkedVJPFunction(_AutogradFunction):
    """Autograd bridge for :func:`chunked_vjp`.

    Kept private: the supported entry point is :func:`chunked_vjp`, which
    validates its arguments before reaching autograd.

    Of the three stages described in the module docstring, this class
    implements the first and the third: :meth:`forward` accumulates the total
    without a graph, and :meth:`backward` re-evaluates each batch to
    accumulate its vector-Jacobian product.

    Stage 2 is deliberately absent. The merit function, and any network it
    contains, belong to the caller's ``loss.backward()``. Autograd completes
    that before reaching this Function, and supplies the result to
    :meth:`backward` as ``grad_output``.
    """

    @staticmethod
    def forward(ctx, batch_fn, setup_fn, batches, *params):
        """Accumulate the total over batches without building a graph.

        Args:
            ctx: Autograd context.
            batch_fn: Maps one batch to its contribution to the total.
            setup_fn: Called before each batch, or None.
            batches: The batches to reduce over.
            *params: Tensors to accumulate gradients for.

        Returns:
            torch.Tensor: The summed contributions of every batch.
        """
        # Captured before the loop so stage 3 can replay the same draws. CUDA
        # is touched only when it is *already* initialised: get_rng_state_all()
        # initialises it otherwise, paying context setup and device memory on
        # a run that never leaves the CPU.
        ctx.rng_state = torch.get_rng_state()
        ctx.cuda_rng_state = (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available() and torch.cuda.is_initialized()
            else None
        )

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
        # Detached copy for the consistency check in backward. Cloned rather
        # than aliased: ``total`` is handed to the caller, and an in-place edit
        # on their side would otherwise move this reference with it.
        ctx.stage1_total = total.detach().clone()
        # Tensors go through save_for_backward instead of being assigned to ctx
        # directly. This lets autograd track their versions and detect a param
        # mutated between the two passes, which would otherwise produce
        # gradients evaluated at the wrong point.
        ctx.save_for_backward(*params)

        return total

    @staticmethod
    @_once_differentiable
    def backward(ctx, grad_output):
        """Re-evaluate each batch and accumulate its VJP.

        ``grad_output`` is dL/d(total). Because the total is a plain sum of
        the batches, d(total)/d(batch_i) is 1, so the same ``grad_output``
        is the correct incoming adjoint for every batch. This is the basis on
        which chunking is valid, and the reason the reduction must be
        additive.

        Args:
            ctx: Autograd context.
            grad_output (torch.Tensor): Gradient of the merit function with
                respect to the total.

        Returns:
            tuple: Gradients matching ``forward``'s signature. The three
            leading ``None`` entries correspond to its non-tensor arguments.
        """
        params = ctx.saved_tensors
        param_grads = [torch.zeros_like(p) for p in params]
        # A param that never receives a VJP from any batch would otherwise be
        # reported as having a zero gradient, which is indistinguishable from a
        # genuine zero. Track connectivity and raise instead.
        connected = [False] * len(params)
        # Running sum of what stage 3 actually evaluated, compared against the
        # stage-1 total once the loop finishes.
        recomputed = None

        # Stage 3 must draw the same random numbers as stage 1. Restoring
        # the global generators covers a batch_fn that samples through
        # them, which optiland's own distributions do under the torch
        # backend. It cannot cover a generator the caller holds and
        # advances, nor numpy or Python random; the consistency check
        # below is the backstop for those.
        outer_rng_state = torch.get_rng_state()
        outer_cuda_rng_state = (
            torch.cuda.get_rng_state_all() if ctx.cuda_rng_state else None
        )
        try:
            torch.set_rng_state(ctx.rng_state)
            if ctx.cuda_rng_state:
                torch.cuda.set_rng_state_all(ctx.cuda_rng_state)

            # enable_grad() is required for correctness. It is not a
            # performance choice: @once_differentiable runs this method in
            # no_grad, so without it batch_fn builds no graph and
            # autograd.grad fails with "element 0 of tensors does not
            # require grad".
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
                            "pass. A batch is a subset of the rays, and its rays "
                            "may contribute anywhere in the total, so every batch "
                            "must return the whole total. Returning only the part "
                            "a batch happens to touch is the usual cause."
                        )

                    detached = contribution.detach()
                    recomputed = (
                        detached.clone()
                        if recomputed is None
                        else recomputed + detached
                    )

                    vjps = torch.autograd.grad(
                        contribution,
                        params,
                        grad_outputs=grad_output,
                        retain_graph=False,  # frees this batch's graph
                        allow_unused=True,
                    )

                    for i, vjp in enumerate(vjps):
                        if vjp is not None:
                            param_grads[i] += vjp.detach()
                            connected[i] = True
        finally:
            # Leave the caller's stream where it was found.
            torch.set_rng_state(outer_rng_state)
            if outer_cuda_rng_state:
                torch.cuda.set_rng_state_all(outer_cuda_rng_state)

        # Checked before connectivity: if the two passes disagree the gradients
        # are wrong whichever params were reached, and this names the cause.
        if recomputed is not None:
            tol = _consistency_tolerance(ctx.stage1_total, len(ctx.batches))
            if not torch.allclose(
                recomputed, ctx.stage1_total, rtol=0.0, atol=tol, equal_nan=True
            ):
                delta = (recomputed - ctx.stage1_total).abs()
                finite_delta = delta[torch.isfinite(delta)]
                detail = (
                    f"by up to {float(finite_delta.max()):.3e} (tolerance {tol:.3e})"
                    if finite_delta.numel()
                    else "in the placement of non-finite values"
                )
                raise RuntimeError(
                    "batch_fn did not reproduce the forward pass: re-evaluating "
                    f"the batches gave a total differing {detail}. The gradient "
                    "would be taken through a different quantity from the one "
                    "that was accumulated. Usual causes: batch_fn draws fresh "
                    "random numbers or reads mutable global state; a generator "
                    "advanced across calls; or a batch consumed by the forward "
                    "pass, such as a one-shot iterator."
                )

        unused = [i for i, ok in enumerate(connected) if not ok]
        if unused:
            raise RuntimeError(
                f"No gradient reached params at position(s) {unused}. They are "
                "not connected to the graph built by batch_fn, so their "
                "gradients would silently be zero. Check that batch_fn uses "
                "these tensors, and that setup_fn re-injects them into the "
                "system on every call instead of reading them out as plain "
                "values."
            )

        # Clear the reference cycle so the batches and closures can be freed.
        ctx.batch_fn = None
        ctx.setup_fn = None
        ctx.batches = None
        ctx.stage1_total = None

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
    derived from it and the gradients land on ``params`` as usual. The module
    docstring describes the staging that makes this possible.

    Args:
        batch_fn: Maps one element of ``batches`` to that batch's
            contribution to the total. Must return the **same shape** on
            every call: a batch's rays may contribute anywhere in the total,
            so the contribution has the shape of the whole total.
        batches: The batches to reduce over, for example index slices into a
            pre-generated set of pupil samples. Each batch is used once per
            pass, so it must survive being used twice. An index slice or
            array works; a one-shot iterator does not. Passing slices keeps
            the result invariant to batch size; per-batch random seeds would
            not.
        params: Tensors to accumulate gradients for. Each must be reachable
            from the graph ``batch_fn`` builds.
        setup_fn: Called before every batch in both passes, for systems that
            need ``params`` re-injected before tracing. Must be idempotent.

    Returns:
        torch.Tensor: The sum of every batch's contribution.

    Raises:
        RuntimeError: If torch is unavailable, if the active backend is not
            torch, if ``batch_fn`` does not reproduce the forward pass, or if
            any entry of ``params`` is unreachable from the graph built by
            ``batch_fn``.
        ValueError: If ``batches`` is empty or contains an iterator, if
            ``params`` is empty or holds a tensor that does not require grad,
            or if ``batch_fn`` returns inconsistent shapes.

    Warning:
        Valid only for reductions that are genuinely **additive** over rays,
        and only for a deterministic ``batch_fn``. A ``batch_fn`` that does
        not reproduce the forward pass raises. A non-additive reduction does
        not, and cannot: it yields silently incorrect gradients. See the
        module docstring for why the two differ.

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
    # silently producing zero gradients instead of an error.
    batches = list(batches)
    if not batches:
        raise ValueError("batches is empty; nothing to reduce over.")

    # Each batch is used once per pass, so a batch the forward pass consumes
    # leaves the backward pass nothing to re-evaluate. Rejecting Iterator
    # instances catches the common forms -- iter(), generators, map, zip --
    # up front, with a message naming the fix. It cannot catch every
    # consumable object; the stage-1/stage-3 comparison in backward is the
    # backstop for the rest.
    consumable = [i for i, b in enumerate(batches) if isinstance(b, Iterator)]
    if consumable:
        raise ValueError(
            f"batches at position(s) {consumable} are iterators, which the "
            "forward pass consumes. The backward pass would then re-evaluate "
            "them as empty and the gradient would come back zero. Each batch "
            "is used once per pass, so pass something re-usable: an index "
            "slice, a list, or an array."
        )

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
