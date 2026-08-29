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


# Subclassing happens when this module is imported, so the base class has to
# exist even where torch does not. ``optiland.ml`` is imported unconditionally
# and the numpy path has to keep working, hence the fallback to ``object``.
# The same pattern is used in optiland/ml/wrappers.py.
_AutogradFunction = torch.autograd.Function if torch is not None else object


class _ChunkedVJPFunction(_AutogradFunction):
    """Autograd bridge for :func:`chunked_vjp`.

    Kept private: the supported entry point is :func:`chunked_vjp`, which
    validates its arguments before anything reaches autograd.
    """

    @staticmethod
    def forward(ctx, batch_fn, setup_fn, batches, *params):
        """Accumulate the total over batches.

        Args:
            ctx: Autograd context.
            batch_fn: Maps one batch to its contribution to the total.
            setup_fn: Called before each batch, or None.
            batches: The batches to reduce over.
            *params: The tensors to differentiate with respect to. Passed
                positionally so that autograd records them as inputs to this
                Function, which is what lets gradients flow back to them.

        Returns:
            torch.Tensor: The total.
        """
        raise NotImplementedError

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
    return _ChunkedVJPFunction.apply(batch_fn, setup_fn, batches, *params)
