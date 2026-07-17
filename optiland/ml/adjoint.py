"""Adjoint Back-Propagation Module

This module contains the AdjointRenderingFunction and AdjointOpticalSystemModule
for memory-efficient differentiable ray tracing and image rendering.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

from optiland.ml.wrappers import OpticalSystemModule

if TYPE_CHECKING:
    from collections.abc import Callable

    from optiland.optic import Optic
    from optiland.optimization.problem import OptimizationProblem


class AdjointRenderingFunction(torch.autograd.Function):
    """
    A custom PyTorch autograd.Function that implements the Adjoint
    Back-Propagation method for image rendering.

    This function decouples the forward ray tracing pass from the network's
    backward pass, chunking the backward pass to maintain an O(1) memory footprint
    with respect to the total number of rays.
    """

    @staticmethod
    def forward(
        ctx,
        render_batch_fn: Callable,
        update_optics_fn: Callable,
        batches: list,
        *params,
    ):
        """
        Forward pass: Renders the image by accumulating chunks without tracking
        gradients.

        Args:
            ctx: The autograd context.
            render_batch_fn: Callable that takes a batch and returns a rendered
                image chunk.
            update_optics_fn: Callable that rebuilds optic intermediate tensors
                from params.
            batches: Iterable of batches (e.g., ray definitions) ensuring
                deterministic tracing.
            params: Differentiable optical parameters.

        Returns:
            torch.Tensor: The accumulated rendered image.
        """
        image = 0.0
        # Forward pass is executed without building a computational graph
        with torch.no_grad():
            # Update the optic to ensure the correct values are traced
            update_optics_fn()

            # [CHUNKING / RAY BATCHING]: We loop through the batches of rays
            # sequentially. This prevents memory from scaling with the total
            # number of rays, ensuring O(1) footprint.
            for batch in batches:
                image += render_batch_fn(batch)

        # Save context for backward pass
        ctx.render_batch_fn = render_batch_fn
        ctx.update_optics_fn = update_optics_fn
        ctx.batches = batches
        ctx.save_for_backward(*params)

        return image

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Re-evaluates chunks with gradients enabled and computes VJPs.

        Args:
            ctx: The autograd context.
            grad_output: The back-propagated gradient from the loss w.r.t the image.

        Returns:
            Gradients w.r.t to the inputs of the forward method.
        """
        render_batch_fn = ctx.render_batch_fn
        update_optics_fn = ctx.update_optics_fn
        batches = ctx.batches
        params = ctx.saved_tensors

        # Initialize gradient accumulators for optical parameters
        param_grads = [torch.zeros_like(p) for p in params]

        # Enable gradients explicitly for the adjoint re-evaluation
        with torch.enable_grad():
            # [CHUNKING / RAY BATCHING]: We iterate over the chunks sequentially again.
            for batch in batches:
                # Rebuild the intermediate tensors in the optic for this chunk's graph
                update_optics_fn()

                # Re-evaluate the chunk, building a temporary local graph
                outputs = render_batch_fn(batch)

                # [SEPARABILITY PROPERTY]: Compute Vector-Jacobian Product (VJP)
                # for this chunk. `grad_output` is the "Derivative Image" (Delta I)
                # coming from the Neural Network. By passing it as `grad_outputs`,
                # we decouple the hardware (ray-tracing) backprop from the
                # software (CNN) backprop.
                vjps = torch.autograd.grad(
                    outputs,
                    params,
                    grad_outputs=grad_output,
                    retain_graph=False,  # chunk graph is instantly destroyed here
                    allow_unused=True,
                )

                # Accumulate the computed gradients
                for i, vjp in enumerate(vjps):
                    if vjp is not None:
                        param_grads[i] += vjp.detach()

        # Free context references to avoid cycles
        ctx.render_batch_fn = None
        ctx.update_optics_fn = None
        ctx.batches = None

        # Return gradients. None is returned for non-tensor inputs
        return (None, None, None, *param_grads)


class AdjointOpticalSystemModule(OpticalSystemModule):
    """
    A PyTorch nn.Module that wraps an Optiland OptimizationProblem and uses
    the Adjoint Back-Propagation method for memory-efficient differentiable rendering.

    This module should be used instead of OpticalSystemModule when rendering
    high-resolution images that require tracing millions of rays, as it avoids
    the severe O(N*H*W) memory bottleneck of standard Autograd.
    """

    def __init__(self, optic: Optic, problem: OptimizationProblem):
        """
        Initializes the AdjointOpticalSystemModule.

        Args:
            optic (Optic): The optical system definition.
            problem (OptimizationProblem): The optimization problem defining variables.
        """
        # Initialize the base module. We do not use the objective_fn here
        # since forward() returns an image, not a scalar loss.
        super().__init__(optic, problem, objective_fn=None)

    def forward(self, batches: list, render_batch_fn: Callable) -> torch.Tensor:
        """
        Defines the forward pass for Adjoint rendering.

        Args:
            batches (list): A list of deterministic batch definitions
                (e.g., explicit ray coordinates or random seeds). Must be chunked
                appropriately to fit within GPU/CPU memory limits.
            render_batch_fn (Callable): A function that accepts a single batch
                from `batches`, traces the rays through the current optical system,
                and returns a projected image chunk tensor.

        Returns:
            torch.Tensor: The final accumulated rendered image tensor.
        """
        if torch is None:
            raise RuntimeError(
                "AdjointOpticalSystemModule requires the 'torch' package."
            )

        def update_optics_fn():
            # 1. Synchronize the nn.Parameter values with the Optiland problem
            # variables.
            self._sync_params_to_problem()
            # 2. Update dependent properties within the optical system
            self.problem.update_optics()

        # 3. Call the custom autograd function to render the image
        image = AdjointRenderingFunction.apply(
            render_batch_fn, update_optics_fn, batches, *self.params
        )

        return image
