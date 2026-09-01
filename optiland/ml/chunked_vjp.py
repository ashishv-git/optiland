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

A *batch* is a subset of the rays. Its rays may contribute to any element of
the total tensor, so a batch's contribution matches the shape of the whole
total. For example, when the total is a rendered image, a batch's
contribution is that whole image, dim and sparsely sampled. A tile of the
image, fully illuminated, would be the wrong shape.

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
# The fallback is needed because a class statement evaluates its base class
# when the file is imported, so that name has to exist either way. optiland
# supports a numpy-only install. Once ml/__init__.py exports chunked_vjp,
# importing optiland.ml will run this file, so on a machine without torch the
# class statement below would fail if this assignment had no fallback.
#
# A class built on ``object`` has no apply() and cannot be used, but the
# import succeeds and a numpy-only user is unaffected. wrappers.py guards
# OpticalSystemModule the same way, so this is the established approach here.
_AutogradFunction = torch.autograd.Function if torch is not None else object


class _ChunkedVJP(_AutogradFunction):
    """Autograd bridge for :func:`chunked_vjp`.

    Kept private: the supported entry point is :func:`chunked_vjp`, which
    validates its arguments before anything reaches autograd.
    """

    @staticmethod
    def forward(ctx, batch_fn, setup_fn, batches, *params):
        """Accumulate the total over the batches without building a graph.

        Args:
            ctx: Autograd context.
            batch_fn: Maps one batch to its contribution to the total.
            setup_fn: Called before each batch, or None.
            batches: The batches to reduce over.
            *params: The tensors to differentiate with respect to. Passed
                positionally so that autograd records them as inputs to this
                Function, which is what lets gradients flow back to them.

        Returns:
            torch.Tensor: The total, the sum of every batch's contribution.
        """
        total = None
        # Disables gradient recording for the accumulation. Autograd already
        # runs forward() with gradients off when this Function is reached
        # through apply(), so on that path the context manager changes
        # nothing. It is kept for two reasons: stage 1 is a no-grad forward
        # pass by definition, and calling forward() directly bypasses apply()
        # and would otherwise build the very graph this module avoids.
        with torch.no_grad():
            for batch in batches:
                if setup_fn is not None:
                    setup_fn()
                contribution = batch_fn(batch)
                if total is None:
                    # Cloned because the following batches are added into it.
                    # batch_fn may hand back a tensor the caller still holds,
                    # and accumulating into that would change a value on their
                    # side.
                    total = contribution.clone()
                else:
                    # Added in place so that a batch returning a different
                    # shape raises here. Out-of-place addition broadcasts
                    # instead, turning a (3,) total plus a (3, 1) contribution
                    # into a (3, 3) one with no error. Returning only the part
                    # of the total a batch touches is the mistake this catches.
                    total += contribution

        # The backward pass replays every batch, so it needs these three
        # again. None of them is a tensor, so they are assigned to ctx
        # directly.
        ctx.batch_fn = batch_fn
        ctx.setup_fn = setup_fn
        ctx.batches = batches
        # The tensors go through save_for_backward, which records the version
        # of each one. A param modified in place between the two passes then
        # raises when backward reads it back. Assigning them to ctx would skip
        # that check and differentiate at a point the forward pass never
        # evaluated.
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
