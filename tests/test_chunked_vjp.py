"""Gate tests for chunked VJP accumulation.

Scientific purpose:
- verify chunked accumulation reproduces plain non-chunked autograd, and
- verify the result does not depend on how the rays were chunked,
- while the autograd graph stays O(1) in the number of chunks.

The first two establish correctness: chunking is a memory strategy, not a
numerical approximation, so the result must be unchanged. The third establishes
the benefit, and is asserted on graph node counts; process memory is too
noisy to regress against.

Equivalence is checked against a reference that never chunks, so a bug in the
chunking logic cannot hide by being present on both sides.
"""

from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.ml.chunked_vjp import chunked_vjp

from .nr_implicit_test_utils import backend_state, count_autograd_nodes

torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def _torch_backend_float64_cpu():
    with backend_state("torch", precision="float64"):
        yield


# --------------------------------------------------------------------------
# A reduction with no optics in it.
#
# This involves no rays, no surfaces and no image. The primitive is not a
# renderer, and exercising it without an optical system present demonstrates
# that directly.
# --------------------------------------------------------------------------

N_SAMPLES = 60
OUTPUT_SIZE = 4


def _make_samples():
    """Fixed sample set standing in for pre-generated pupil samples."""
    generator = torch.Generator().manual_seed(0)
    return torch.randn(N_SAMPLES, OUTPUT_SIZE, dtype=torch.float64,
                       generator=generator)


def _reduce(samples, theta):
    """A nonlinear but additive reduction over samples.

    Additive over the sample axis -- ``_reduce(all) == sum(_reduce(part))`` --
    which is exactly the precondition chunked_vjp requires. Nonlinear in
    ``theta`` so the gradient is not trivially constant.
    """
    return torch.tanh(samples * theta).sum(dim=0)


def _chunks(total, size):
    """Index slices covering ``total`` items in chunks of ``size``."""
    return [slice(i, min(i + size, total)) for i in range(0, total, size)]


def test_gradient_matches_non_chunked_autograd():
    """Chunking changes the memory profile, not the result."""
    samples = _make_samples()

    theta_ref = torch.tensor([0.7] * OUTPUT_SIZE, dtype=torch.float64,
                             requires_grad=True)
    theta_chunked = theta_ref.detach().clone().requires_grad_(True)

    # Reference: one graph over every sample at once.
    _reduce(samples, theta_ref).sum().backward()

    # Chunked: same reduction, graph built and freed one chunk at a time.
    total = chunked_vjp(
        lambda sl: _reduce(samples[sl], theta_chunked),
        _chunks(N_SAMPLES, 8),
        params=[theta_chunked],
    )
    total.sum().backward()

    # float64 accumulation over 8 chunks; the only difference admissible here
    # is summation order, which is well below this tolerance.
    assert torch.allclose(theta_chunked.grad, theta_ref.grad, rtol=0, atol=1e-14)


@pytest.mark.parametrize("chunk_size", [1, 7, 8, N_SAMPLES, N_SAMPLES + 5])
def test_result_is_invariant_to_chunk_size(chunk_size):
    """Chunk size controls memory use only, and must not affect the result.

    Includes sizes that do not divide the total evenly, a size equal to the
    total (one chunk) and a size larger than it (a short final chunk), which
    are the cases where off-by-one partitioning errors appear.
    """
    samples = _make_samples()

    theta_ref = torch.tensor([0.7] * OUTPUT_SIZE, dtype=torch.float64,
                             requires_grad=True)
    reference = _reduce(samples, theta_ref)
    reference.sum().backward()

    theta = theta_ref.detach().clone().requires_grad_(True)
    total = chunked_vjp(
        lambda sl: _reduce(samples[sl], theta),
        _chunks(N_SAMPLES, chunk_size),
        params=[theta],
    )
    total.sum().backward()

    assert torch.allclose(total, reference, rtol=0, atol=1e-14)
    assert torch.allclose(theta.grad, theta_ref.grad, rtol=0, atol=1e-14)


# Chunk sizes over the same 60 samples, giving 2, 6 and 30 chunks. Shared by
# the pair of graph-size tests below so they compare like with like.
_GRAPH_CHUNK_SIZES = (30, 10, 2)


def _accumulate_naively(samples, theta, chunk_size):
    """Chunk the reduction the way a user would without this primitive.

    Every chunk's graph is retained until backward, because ``total``
    references all of them. This is the baseline for comparison.
    """
    total = None
    for sl in _chunks(N_SAMPLES, chunk_size):
        contribution = _reduce(samples[sl], theta)
        total = contribution if total is None else total + contribution
    return total


def test_naive_accumulation_graph_grows_with_chunk_count():
    """Confirm that the baseline does grow with chunk count.

    Without this, the flatness asserted below could be satisfied by a
    primitive that built no useful graph.
    """
    samples = _make_samples()
    theta = torch.tensor([0.7] * OUTPUT_SIZE, dtype=torch.float64,
                         requires_grad=True)

    node_counts = [
        count_autograd_nodes(_accumulate_naively(samples, theta, size))
        for size in _GRAPH_CHUNK_SIZES
    ]

    assert node_counts == sorted(node_counts), (
        f"Baseline graph did not grow monotonically with chunks: {node_counts}"
    )
    assert node_counts[-1] >= 10 * node_counts[0], (
        f"Baseline growth is too weak to be a meaningful control: {node_counts}"
    )


def test_chunked_graph_size_is_flat_vs_chunk_count():
    """Graph size must not grow with the number of chunks.

    Asserted on autograd node counts; process memory is far too noisy to
    regress against. The forward pass runs under ``no_grad``, so
    the returned tensor carries only what this primitive contributes,
    regardless of chunk count.
    """
    samples = _make_samples()
    theta = torch.tensor([0.7] * OUTPUT_SIZE, dtype=torch.float64,
                         requires_grad=True)

    node_counts = [
        count_autograd_nodes(
            chunked_vjp(
                lambda sl: _reduce(samples[sl], theta),
                _chunks(N_SAMPLES, size),
                params=[theta],
            )
        )
        for size in _GRAPH_CHUNK_SIZES
    ]

    assert len(set(node_counts)) == 1, (
        f"Graph size varied with chunk count: {node_counts}"
    )

    naive_counts = [
        count_autograd_nodes(_accumulate_naively(samples, theta, size))
        for size in _GRAPH_CHUNK_SIZES
    ]
    assert node_counts[-1] < naive_counts[-1], (
        f"At {N_SAMPLES // _GRAPH_CHUNK_SIZES[-1]} chunks the chunked graph "
        f"({node_counts[-1]} nodes) is not smaller than the naive one "
        f"({naive_counts[-1]} nodes)."
    )


# --------------------------------------------------------------------------
# The same claims, through a real optical trace.
# --------------------------------------------------------------------------


def _make_singlet():
    from optiland.materials import Material
    from optiland.optic import Optic

    lens = Optic("Singlet")
    lens.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    lens.surfaces.add(
        index=1,
        radius=70.0,
        thickness=7.0,
        material=Material("N-BK7"),
        is_stop=True,
    )
    lens.surfaces.add(index=2, radius=-70.0, thickness=70.0)
    lens.surfaces.add(index=3)
    lens.set_aperture(aperture_type="EPD", value=25.0)
    lens.fields.set_type("angle")
    lens.fields.add(y=0.0)
    lens.wavelengths.add(value=0.55, is_primary=True)
    return lens


class _FixedDistribution:
    """A pupil sample set that can be sliced.

    Chunking must partition one fixed set of samples. Re-drawing random
    samples per chunk would make the result depend on the chunk size, the
    failure mode ``test_result_is_invariant_to_chunk_size`` detects.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def generate_points(self, num_points):  # noqa: ARG002 - already generated
        return None

    def __getitem__(self, sl):
        return _FixedDistribution(self.x[sl], self.y[sl])


def _pupil_samples(num_points):
    from optiland.distribution import RandomDistribution

    distribution = RandomDistribution(seed=42)
    distribution.generate_points(num_points)
    return _FixedDistribution(distribution.x, distribution.y)


def _trace_and_reduce(lens, distribution):
    """Sum ray coordinates at the image plane.

    Additive over rays and, unlike an image render, free of binning that
    could mask a gradient error behind a discretisation.
    """
    rays = lens.trace(
        Hx=0.0,
        Hy=0.0,
        wavelength=0.55,
        num_rays=len(distribution.x),
        distribution=distribution,
    )
    return torch.stack([rays.x.sum(), rays.y.sum()])


def test_gradient_matches_non_chunked_autograd_through_a_trace():
    """Gradient equivalence through the real ray tracer."""
    be.grad_mode.enable()
    num_rays = 48

    lens_ref = _make_singlet()
    radius_ref = lens_ref.surfaces.surfaces[1].geometry.radius
    radius_ref.requires_grad_(True)
    _trace_and_reduce(lens_ref, _pupil_samples(num_rays)).sum().backward()

    lens = _make_singlet()
    radius = lens.surfaces.surfaces[1].geometry.radius
    radius.requires_grad_(True)
    samples = _pupil_samples(num_rays)

    total = chunked_vjp(
        lambda sl: _trace_and_reduce(lens, samples[sl]),
        _chunks(num_rays, 16),
        params=[radius],
    )
    total.sum().backward()

    assert radius.grad is not None
    assert torch.allclose(radius.grad, radius_ref.grad, rtol=1e-10, atol=1e-12)
