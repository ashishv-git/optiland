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

Placeholder beyond this point. The remaining sections are added as the code
they describe is written.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    import torch
except ImportError:  # pragma: no cover - only runs when torch is not installed
    torch = None

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


def _check_contribution_shape(contribution, expected, expected_desc: str) -> None:
    """Raise if one batch's contribution is not shaped like the total.

    Called from both passes, which compare against different tensors: the
    forward pass against the total the first batch established, the backward
    pass against the incoming gradient. The message is the same either way, so
    it lives here instead of being written out twice.

    Both comparisons are against another value the same ``batch_fn`` produced,
    so this establishes consistency and not correctness. A ``batch_fn``
    returning the same wrong shape on every call is self-consistent and passes:
    four equal tiles of an image sum to a tile-shaped total without error.

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
            f"{tuple(expected.shape)}. The total is the plain sum of the batch "
            f"contributions, so every contribution has the shape of the whole "
            f"total. Returning only the part of the total a batch's own rays "
            f"reach is the usual cause."
        )


class _ChunkedVJP(_AutogradFunction):
    """Autograd bridge for :func:`chunked_vjp`.

    Kept private: the supported entry point is :func:`chunked_vjp`, which
    validates its arguments before anything reaches autograd.
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

        # no_grad() disables gradient recording for the accumulation. Autograd
        # already runs forward() with gradients off when this Function is reached
        # through apply(), so on that path this changes nothing. It is kept
        # for the case it does cover: calling forward() directly leaves
        # gradients enabled, and the accumulation would then build the graph
        # this method exists to avoid.
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
        # The three above are plain ctx attributes; params are saved with
        # save_for_backward instead, which records each tensor's version
        # counter -- a number torch increments on every in-place modification.
        # Reading them back as ctx.saved_tensors compares the counters and
        # raises RuntimeError if any has changed: "one of the variables needed
        # for gradient computation has been modified by an inplace operation".
        # Plain attributes would skip that comparison, and backward would then
        # compute gradients from parameter values the forward pass never used.
        ctx.save_for_backward(*params)

        return total

    @staticmethod
    def backward(ctx, grad_output):
        """Re-evaluate each batch and accumulate its vector-Jacobian product.

        Args:
            ctx: Autograd context.
            grad_output (torch.Tensor): Gradient of the merit function with
                respect to the total.

        Returns:
            tuple: One gradient per argument ``forward`` received, in the same
            order. The leading entries are ``None`` because ``batch_fn``,
            ``setup_fn`` and ``batches`` are not differentiable.
        """
        raise NotImplementedError


def chunked_vjp(
    batch_fn: Callable[[Any], torch.Tensor],
    batches: Sequence[Any],
    params: Sequence[torch.Tensor],
    *,
    setup_fn: Callable[[], None] | None = None,
) -> torch.Tensor:
    """Differentiable sum of ``batch_fn`` over ``batches``, with bounded graph size.

    Args:
        batch_fn: Maps one element of ``batches`` to that batch's contribution
            to the total.
        batches: The batches of rays to reduce over. A typical choice is to
            generate all the rays once, then pass slices into that array.
        params: The tensors to differentiate with respect to.
        setup_fn: Called before every batch. Use it when the system being
            traced keeps its own copy of the parameters and needs the current
            values before each trace.

    Returns:
        torch.Tensor: The total, the sum of every batch's contribution.
    """
    # TODO: docstring sections still to add, each alongside the code that
    # makes it true -- a body paragraph on graph lifetime, Raises, Warning,
    # Example, and the per-argument constraints (same shape on every call,
    # batches re-usable, params requiring grad and reachable, setup_fn called
    # in both passes and therefore needing to be idempotent). The finished
    # version is on feat/chunked-vjp-reference if a reference is wanted.
    #
    # TODO: when the entry checks are written, reject a bare torch.Tensor
    # passed as params. A tensor is itself a sequence, so params=radius
    # instead of params=[radius] iterates it into row views. Every other
    # guard lets that through: the views require grad, so the requires_grad
    # check passes, and they are genuinely in the graph, so the connectivity
    # check passes. Gradients are then computed against temporaries and
    # discarded, leaving the caller's tensor with .grad still None and their
    # optimiser doing nothing. Detect it with is_leaf, which is False for
    # every element in that case.
    #
    # TODO: when backward() is written, put a comment beside the
    # torch.autograd.grad call explaining why it is used instead of
    # .backward(). A custom Function must return gradients for autograd to
    # apply; calling .backward() there applies them as well, and the result
    # is silently doubled. autograd.grad returns them, and it is what
    # requires the explicit params list. Without the comment the next reader
    # sees a longer form of something .backward() would do in one line.
    #
    # Each tensor is passed as its own argument to apply(). Autograd works out
    # what this Function depends on by looking at the arguments it receives,
    # and it does not look inside a list. Passing params as a single list
    # argument leaves the tensors unregistered, the returned total does not
    # require grad, and .backward() then fails with "element 0 of tensors does
    # not require grad".
    return _ChunkedVJP.apply(batch_fn, setup_fn, batches, *params)
