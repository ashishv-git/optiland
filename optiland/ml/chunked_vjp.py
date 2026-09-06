"""Chunked vector-Jacobian product accumulation.

Differentiating a ray trace requires memory proportional to the number of
rays, because reverse-mode autograd retains the whole graph until the
backward pass runs. Many optical quantities are *reductions* over rays -- a
rendered image, an irradiance map -- each accumulating one contribution per
ray into a single total. Differentiating such a reduction over enough rays
produces an autograd graph that exceeds device memory.

The loss is computed from ``params`` in two steps::

    params --ray trace and reduce--> total --merit function--> loss

``params`` are the tensors to differentiate with respect to, and ``total`` is
the tensor the reduction accumulates into. The merit function is any
scalar-valued function of the total: for example a least-squares comparison
against a target, or a neural network computing a perceptual loss.

A *batch* is a subset of the rays. The total is the plain sum of the batch
contributions, so every contribution has the shape of the whole total. For
example, when the total is a rendered image, a batch's contribution is that
whole image, dim and sparsely sampled. A tile of the image, fully
illuminated, would be the wrong shape. This holds even when a batch's rays
reach only part of the total: the contribution is still the whole image, zero
outside the region those rays cover.

``chunked_vjp`` passes each batch to ``batch_fn`` unchanged and never inspects
it. It places no restriction on what ``batch_fn`` computes. Any reduction over
rays can therefore be chunked, provided it is additive.

Summing the contributions in an ordinary loop would not lower peak graph
memory. Each batch's graph stays reachable from the total, so autograd keeps
them all until the backward pass runs. ``chunked_vjp`` frees each batch's
graph before building the next, so only one exists at a time. Peak graph
memory is then set by the largest graph a single batch produces, and does not
grow with the number of batches.

Every ray is traced twice: once in the forward pass to accumulate the total,
and once in the backward pass to differentiate it. Batch size does not change
how many times each ray is traced. Per-batch work runs once for each batch in
each pass, so ``setup_fn`` is called twice for every batch. Smaller batches
lower peak graph memory and raise the number of those calls.

How the computation is split
----------------------------
The graph is cut at the total. Writing ``L`` for the merit function, the
chain rule gives::

    dL/d(params) = dL/d(total) * d(total)/d(params)

The two factors can be evaluated in separate passes. dO calls this the
**separability property**, and uses it to split the work into three stages
(section II-D):

1. **Forward, no autograd.** :func:`chunked_vjp` accumulates the total over
   the batches with gradients disabled, so no graph is built.
2. **Forward and backward on the merit function.** The caller applies the
   merit function to the total and calls ``.backward()`` on the resulting
   scalar loss. Autograd propagates back through the merit function,
   including any network inside it, and arrives at the total carrying
   ``dL/d(total)``, a tensor of the same shape.
3. **Forward and backward on the reduction.** One batch at a time,
   :func:`chunked_vjp` re-evaluates ``batch_fn`` with gradients enabled,
   rebuilding the graph from ``params`` to that batch's contribution. That
   graph encodes ``d(contribution)/d(params)``, and its vector-Jacobian
   product with ``dL/d(total)`` is the batch's share of ``dL/d(params)``,
   which is accumulated into the parameter gradients. The graph is freed
   before the next batch is built.

Stage 1 runs no backward pass, and the total it returns is an ordinary
autograd tensor. Stage 2 is the caller's own code: they apply the merit
function to that total and call ``.backward()`` on the loss. Autograd reaches
:func:`chunked_vjp` during that call and runs stage 3. Stage 3 needs only
``dL/d(total)`` from stage 2, and never evaluates the merit function again.

Preconditions
-------------
Two conditions must hold for the gradients to be correct: the reduction must
be additive over rays, and ``batch_fn`` must be reproducible. Violating
either gives incorrect gradients. ``chunked_vjp`` cannot check additivity, so
a non-additive reduction fails silently. Reproducibility could be checked by
comparing what stage 1 and stage 3 compute, but this version does not compare
them. It raises when a contribution changes shape between the passes, and does
not detect a ``batch_fn`` that returns different values.

.. warning::
   The reduction must be genuinely **additive** over rays. The total must be the
   plain sum of the batch contributions. This ensures stage 3 applies the same
   ``dL/d(total)`` to every batch.

   This precondition fails for ``max``. It also fails for any quantity
   normalized by something derived from all rays, such as a centroid reference
   or a count of surviving rays.

   :func:`chunked_vjp` cannot detect non-additive reductions. The module
   never inspects the batches, so it cannot regroup the rays to check if a
   different batching produces a different total. The contributions are just
   tensors; they do not record which operation produced them. Verifying
   additivity is the caller's responsibility.

.. warning::
   ``batch_fn`` must be **reproducible** even if it includes stochastic
   operations. Stage 3 re-evaluates ``batch_fn``. The resulting contribution
   must exactly match the contribution computed in stage 1. Otherwise, the
   gradient is computed for a different output than the one used in the merit
   function.

   Stochastic operations like ray generation or scattering require explicit
   state management. For example, explicitly setting the random seed before
   processing each batch guarantees that these operations yield the exact same
   values in both passes.

   The batch object must persist across passes. Do not pass a one-shot
   iterator. Instead, generate all the rays once, then pass slices into that
   array.

Relation to implicit differentiation
------------------------------------
Chunked accumulation composes with the implicit differentiation used by the
Newton-Raphson solvers (see the Implicit Differentiation developer guide).
It does not replace it. Implicit differentiation bounds how deep the graph
is per ray. Chunked accumulation bounds how wide the graph is across rays.

Reference
---------
Wang, Chen and Heidrich, "dO: A Differentiable Engine for Deep Lens Design of
Computational Imaging Systems", IEEE Transactions on Computational Imaging,
2022. Section II-D introduces the separability property and calls the
three-stage procedure adjoint back-propagation.

Ashish Verma, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    import torch
except ImportError:  # pragma: no cover - only runs when torch is not installed
    torch = None

import optiland.backend as be

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


# Pick the base class that _ChunkedVJP below will inherit from:
# torch.autograd.Function when torch is installed, and plain ``object`` when
# it is not.
#
# A class statement is executed when the file is imported, and executing it
# evaluates the base class named in its parentheses. Importing this file
# therefore runs ``class _ChunkedVJP(_AutogradFunction):`` below and looks up
# ``_AutogradFunction`` at that moment, so the name has to resolve whether or
# not torch is installed. optiland supports a numpy-only install, and once
# ml/__init__.py imports chunked_vjp from here, importing optiland.ml runs
# this file. Naming ``torch.autograd.Function`` directly in the class
# statement would raise AttributeError on a numpy-only install.
#
# With ``object`` as the base class, _ChunkedVJP has no apply() and cannot be
# used, but the import succeeds and a numpy-only user is unaffected.
# wrappers.py guards OpticalSystemModule the same way, so this is the
# established approach here.
_AutogradFunction = torch.autograd.Function if torch is not None else object

# Pick the decorator that backward() below will carry. once_differentiable
# makes a second differentiation raise instead of returning a wrong answer;
# this module supports first-order gradients only. It needs the same fallback
# as _AutogradFunction above, because a decorator is applied when the class
# body runs, which happens as the file is imported.
if torch is not None:
    _once_differentiable = torch.autograd.function.once_differentiable
else:  # pragma: no cover - only runs when torch is not installed

    def _once_differentiable(fn):
        return fn


def _check_contribution_shape(contribution, expected, expected_desc: str) -> None:
    """Raise if a batch's contribution shape differs from the expected shape.

    Both passes call this function to share the error message formatting. The
    forward pass compares the contribution against the total established by the
    first batch. The backward pass compares it against the incoming gradient.

    Both passes compare the contribution against a value produced by the same
    ``batch_fn``. Therefore, this check establishes consistency rather than
    correctness. If ``batch_fn`` consistently returns the wrong shape on every
    call, it will pass this check. For example, if the true total is shape
    (100, 100), but ``batch_fn`` consistently returns shape (50, 50) on every
    call, it will not raise an error.

    Args:
        contribution (torch.Tensor): What ``batch_fn`` returned for one batch.
        expected (torch.Tensor): The tensor whose shape it has to match.
        expected_desc (str): Names ``expected`` in the error message.

    Raises:
        ValueError: If the two shapes differ.
    """
    if contribution.shape != expected.shape:
        raise ValueError(
            f"batch_fn must return the same shape on every call. It returned "
            f"{tuple(contribution.shape)}, but {expected_desc} has shape "
            f"{tuple(expected.shape)}. The total is the sum of the batch "
            f"contributions, so every contribution must match the shape of the "
            f"total."
        )


class _ChunkedVJP(_AutogradFunction):
    """Autograd bridge for :func:`chunked_vjp`.

    This class is private. The public entry point is :func:`chunked_vjp`.
    That function validates the arguments and then calls
    ``_ChunkedVJP.apply()``.

    This class implements stage 1 and stage 3 from the module docstring.
    :meth:`forward` accumulates the total without building a graph.
    :meth:`backward` re-evaluates each batch to accumulate its vector-Jacobian
    product.

    This class does not implement stage 2. The caller evaluates the merit
    function and calls ``loss.backward()``. Autograd propagates the gradient
    through the merit function to compute the derivative of the loss with
    respect to the total tensor. Autograd then passes this incoming gradient
    into our :meth:`backward` method as ``grad_output`` so that stage 3 can
    apply it to every batch.
    """

    @staticmethod
    def forward(
        ctx,
        batch_fn: Callable[[Any], torch.Tensor],
        setup_fn: Callable[[], None] | None,
        batches: Sequence[Any],
        *params: torch.Tensor,
    ) -> torch.Tensor:
        """Accumulate the total over the batches without building a graph.

        Args:
            ctx: Autograd context. Anything :meth:`backward` needs is stored
                on it here.
            batch_fn: Maps one batch to its contribution to the total.
            setup_fn: A callable taking no arguments, run before every batch.
                None when the caller supplied none.
            batches: The subsets of rays to reduce over. Each is passed to
                ``batch_fn`` unchanged.
            *params: The tensors to differentiate with respect to. Passed
                positionally so that autograd records them as inputs to this
                Function. Gradients flow back only to what it records.

        Returns:
            torch.Tensor: The total, the sum of every batch's contribution.
        """
        # None until the first batch arrives. The total takes its shape and
        # dtype from whatever batch_fn returns, neither of which is known
        # before then, so there is nothing to preallocate.
        total = None

        # Accumulate the total without building a computational graph. We do
        # this by wrapping the loop in torch.no_grad(). If forward() is called
        # directly, gradients are enabled by default; explicitly disabling them
        # prevents the graph from being built. When Autograd calls forward()
        # via apply(), gradients are already disabled, making this a safe no-op.
        with torch.no_grad():
            for batch in batches:
                if setup_fn is not None:
                    setup_fn()
                contribution = batch_fn(batch)
                if total is None:
                    # Cloned because the following batches are added into it.
                    # batch_fn may return a tensor the caller still references,
                    # and accumulating into that would modify the caller's own
                    # tensor in place.
                    total = contribution.clone()
                else:
                    # Checked explicitly because the addition below would
                    # accept a contribution that broadcasts into the total's
                    # shape, such as (1, 4) into (3, 4).
                    _check_contribution_shape(
                        contribution, total, "the first batch's contribution"
                    )
                    # Added in place, so the whole reduction writes into the
                    # tensor the clone allocated instead of allocating one
                    # per batch.
                    total += contribution

        # backward re-evaluates batch_fn for every batch, so it needs batch_fn,
        # setup_fn and batches again. None of them is a tensor, so they go on
        # ctx as plain attributes.
        ctx.batch_fn = batch_fn
        ctx.setup_fn = setup_fn
        ctx.batches = batches
        # Save params for the backward pass while enforcing version checking.
        # We do this by passing them to ctx.save_for_backward() instead of
        # storing them as plain attributes on the ctx object.
        # ctx.save_for_backward() records each tensor's version counter; reading
        # the tensors back via ctx.saved_tensors compares these counters and
        # raises a RuntimeError if any tensor was modified in place. If we
        # stored them as plain attributes on the ctx object, we would bypass
        # this check, and the backward pass would silently compute incorrect
        # gradients using parameter values the forward pass never used.
        ctx.save_for_backward(*params)

        return total

    @staticmethod
    @_once_differentiable
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """Re-evaluate each batch and accumulate its vector-Jacobian product.

        The total is the plain sum of the batch contributions. Consequently, the
        derivative of the total with respect to any single contribution is
        exactly 1. This means the gradient of the loss with respect to the
        total (``grad_output``) can be passed directly to every batch without
        modification. The chunking approach is valid due to this property, and
        this property holds only when the reduction is additive.

        Args:
            ctx: The autograd context containing the objects saved by
                :meth:`forward`.
            grad_output: The gradient of the loss with respect to the ``total``
                tensor.

        Returns:
            tuple: A tuple containing one gradient for each argument passed to
                :meth:`forward`, in the exact same order. The first three
                gradients are ``None`` because ``batch_fn``, ``setup_fn``, and
                ``batches`` are not tensors and do not require gradients.

        Raises:
            ValueError: If ``batch_fn`` returns a shape that differs from the
                shape of ``grad_output`` (the gradient of the loss with respect
                to the ``total``).
            RuntimeError: If any tensor in ``params`` is disconnected from the
                computational graph across all batches.
        """
        params = ctx.saved_tensors
        param_grads = [torch.zeros_like(p) for p in params]
        # Track which parameters the computational graph reaches to prevent
        # returning false zeros. We do this by recording a boolean flag per
        # parameter. If a parameter is never reached by any batch, it would
        # otherwise finish with a zero gradient, which is indistinguishable
        # from a genuine mathematical zero. Tracking connectivity allows us to
        # raise a RuntimeError for unreachable parameters after the loop ends.
        connected = [False] * len(params)

        # enable_grad() is required for correctness here, unlike the no_grad()
        # in forward. once_differentiable runs this method with gradients
        # disabled, so without it batch_fn records nothing and autograd.grad
        # fails with "element 0 of tensors does not require grad and does not
        # have a grad_fn".
        with torch.enable_grad():
            for batch in ctx.batches:
                if ctx.setup_fn is not None:
                    # Called inside enable_grad so that a system re-reading
                    # params here connects them to the graph batch_fn builds.
                    # Outside it, the re-injected values would be detached and
                    # no gradient would reach params.
                    ctx.setup_fn()

                contribution = ctx.batch_fn(batch)
                _check_contribution_shape(
                    contribution,
                    grad_output,
                    "grad_output (the gradient of the loss with respect to the total)",
                )

                # autograd.grad returns the gradients; the engine applies
                # whatever this method returns. contribution.backward() would
                # apply them here as well, and every gradient would come out
                # doubled with no error raised. Returning them is also what
                # makes an explicit params list necessary: autograd.grad needs
                # its inputs named, which .backward() would have inferred.
                vjps = torch.autograd.grad(
                    contribution,
                    params,
                    grad_outputs=grad_output,
                    # This batch's graph is freed as its gradient is taken, so
                    # only one exists at a time. It is the line that bounds
                    # the memory.
                    retain_graph=False,
                    # Returns None for a param this batch did not reach,
                    # instead of raising. Some batches legitimately miss a
                    # param; a param that every batch misses is the error, and
                    # connected[] below is what tells the two apart.
                    allow_unused=True,
                )

                for i, vjp in enumerate(vjps):
                    if vjp is not None:
                        param_grads[i] += vjp
                        connected[i] = True

        unused = [i for i, ok in enumerate(connected) if not ok]
        if unused:
            raise RuntimeError(
                f"No gradient reached params at position(s) {unused}. They are "
                "not connected to the graph batch_fn builds, so their gradients "
                "would silently be zero. Check that batch_fn uses these "
                "tensors, and that setup_fn puts them back into the system on "
                "every call instead of reading their values out once."
            )

        # ctx is reachable from the grad_fn of the total, which the caller may
        # still hold, so batch_fn, setup_fn and batches would stay alive with
        # it. batch_fn is typically a closure over the whole pre-generated ray
        # set. Dropping the references frees them when this pass ends instead.
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
    """Differentiable sum of ``batch_fn`` over ``batches``, with bounded graph size.

    The result is an ordinary autograd tensor: call ``.backward()`` on a loss
    derived from it and the gradients land on ``params`` as usual. The module
    docstring describes the staging that makes this possible.

    Args:
        batch_fn: Maps one element of ``batches`` to that batch's contribution
            to the total. Must return the **same shape** on every call, since
            the total is the plain sum of the contributions.
        batches: The batches of rays to reduce over. A typical choice is to
            generate all the rays once, then pass slices into that array.
            Each batch is used once per pass, so it has to survive being used
            twice: an index slice or array works, a one-shot iterator does
            not. Slices also keep the result invariant to batch size, which
            per-batch random seeds would not.
        params: The tensors to differentiate with respect to. Each must
            require grad and be reachable from the graph ``batch_fn`` builds.
            Pass a sequence, so ``params=[radius]`` and never
            ``params=radius``.
        setup_fn: Called before every batch, in both passes. Use it when the
            system being traced keeps its own copy of the parameters and
            needs the current values before each trace. Must be idempotent.

    Returns:
        torch.Tensor: The total, the sum of every batch's contribution.

    Raises:
        RuntimeError: If torch is not installed, if the active backend is not
            torch, or if any entry of ``params`` is unreachable from the graph
            ``batch_fn`` builds.
        ValueError: If ``batches`` is empty; if ``params`` is empty, is a
            single tensor, or holds a tensor that does not require grad; or if
            ``batch_fn`` returns a different shape on different calls.

    Warning:
        Valid only for a reduction that is genuinely **additive** over rays,
        and only for a deterministic ``batch_fn``. Neither condition is
        checked, and violating either yields silently incorrect gradients.
        The module docstring gives the reason each is undetectable.

    Example:
        >>> total = chunked_vjp(
        ...     render_batch,
        ...     [slice(i, i + 10_000) for i in range(0, 1_000_000, 10_000)],
        ...     params=[radius],
        ... )
        >>> loss = criterion(total, target)
        >>> loss.backward()
    """
    if torch is None:
        raise RuntimeError(
            "chunked_vjp requires the 'torch' package. Install PyTorch to use "
            "this function."
        )
    if be.get_backend() != "torch":
        raise RuntimeError(
            "chunked_vjp requires the 'torch' backend, but the active backend "
            f"is '{be.get_backend()}'. This feature accumulates gradients "
            "through PyTorch autograd and has no numpy equivalent; call "
            "optiland.backend.set_backend('torch') first."
        )

    # Copied into a list because both passes iterate it. A generator would be
    # exhausted by the forward pass, leaving the backward pass nothing to
    # re-evaluate, and the gradient would come back zero with no error.
    batches = list(batches)
    if not batches:
        raise ValueError("batches is empty; there is nothing to reduce over.")

    # Tested before params is converted, because converting it is what hides
    # the mistake. A tensor is itself a sequence, so params=radius rather than
    # params=[radius] iterates into row views, and every later check is
    # satisfied: the views require grad, and they are genuinely in the graph
    # batch_fn builds, so backward's connectivity check passes too. Gradients
    # accumulate onto those temporaries and are discarded with them, leaving
    # the caller's own tensor with .grad still None and their optimiser with
    # nothing to step.
    if isinstance(params, torch.Tensor):
        raise ValueError(
            "params must be a sequence of tensors, but a single tensor was "
            "given. Iterating it differentiates with respect to its rows "
            "instead of the tensor itself, and no gradient reaches the tensor "
            "you passed. Write params=[tensor] instead of params=tensor."
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

    # Each tensor is passed as its own argument to apply(). Autograd works out
    # what this Function depends on by looking at the arguments it receives,
    # and it does not look inside a list. Passing params as a single list
    # argument leaves the tensors unregistered, the returned total does not
    # require grad, and .backward() then fails with "element 0 of tensors does
    # not require grad".
    return _ChunkedVJP.apply(batch_fn, setup_fn, batches, *params)
