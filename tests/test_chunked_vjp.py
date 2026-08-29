"""Gate tests for chunked VJP accumulation.

Scientific purpose:
- verify chunked accumulation reproduces plain non-chunked autograd, and
- verify the result does not depend on how the rays were chunked,
- while the autograd graph stays O(1) in the number of batches.

The first two establish correctness: chunking is a memory strategy, not a
numerical approximation, so the result must be unchanged. The third establishes
the benefit, and is asserted on graph node counts; process memory is too
noisy to regress against.

Equivalence is checked against a reference computed without chunking, so a bug
in the chunking logic cannot hide by being present on both sides.
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


def _batches(num_samples, size):
    """Index slices covering ``num_samples`` items in batches of ``size``."""
    return [slice(i, min(i + size, num_samples)) for i in range(0, num_samples, size)]


def test_gradient_matches_non_chunked_autograd():
    """Chunking changes the memory profile, not the result."""
    samples = _make_samples()

    theta_ref = torch.tensor([0.7] * OUTPUT_SIZE, dtype=torch.float64,
                             requires_grad=True)
    theta_chunked = theta_ref.detach().clone().requires_grad_(True)

    # Reference: one graph over every sample at once.
    _reduce(samples, theta_ref).sum().backward()

    # Chunked: same reduction, graph built and freed one batch at a time.
    total = chunked_vjp(
        lambda sl: _reduce(samples[sl], theta_chunked),
        _batches(N_SAMPLES, 8),
        params=[theta_chunked],
    )
    total.sum().backward()

    # float64 accumulation over 8 batches; the only difference admissible here
    # is summation order, which is well below this tolerance.
    assert torch.allclose(theta_chunked.grad, theta_ref.grad, rtol=0, atol=1e-14)


@pytest.mark.parametrize("batch_size", [1, 7, 8, N_SAMPLES, N_SAMPLES + 5])
def test_result_is_invariant_to_batch_size(batch_size):
    """Batch size controls memory use only, and must not affect the result.

    Includes sizes that do not divide the sample count evenly, a size equal to
    it (one batch) and a size larger than it (a short final batch), which
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
        _batches(N_SAMPLES, batch_size),
        params=[theta],
    )
    total.sum().backward()

    assert torch.allclose(total, reference, rtol=0, atol=1e-14)
    assert torch.allclose(theta.grad, theta_ref.grad, rtol=0, atol=1e-14)


# Batch sizes over the same 60 samples, giving 2, 6 and 30 batches. Shared by
# the pair of graph-size tests below so they compare like with like.
_GRAPH_BATCH_SIZES = (30, 10, 2)


def _accumulate_naively(samples, theta, batch_size):
    """Accumulate the reduction the way a user would without this primitive.

    Every batch's graph is retained until backward, because ``total``
    references all of them. This is the baseline for comparison.
    """
    total = None
    for sl in _batches(N_SAMPLES, batch_size):
        contribution = _reduce(samples[sl], theta)
        total = contribution if total is None else total + contribution
    return total


def test_naive_accumulation_graph_grows_with_batch_count():
    """Confirm that the baseline does grow with batch count.

    Without this, the flatness asserted below could be satisfied by a
    primitive that built no useful graph.
    """
    samples = _make_samples()
    theta = torch.tensor([0.7] * OUTPUT_SIZE, dtype=torch.float64,
                         requires_grad=True)

    node_counts = [
        count_autograd_nodes(_accumulate_naively(samples, theta, size))
        for size in _GRAPH_BATCH_SIZES
    ]

    assert node_counts == sorted(node_counts), (
        f"Baseline graph did not grow monotonically with batches: {node_counts}"
    )
    assert node_counts[-1] >= 10 * node_counts[0], (
        f"Baseline growth is too weak to be a meaningful control: {node_counts}"
    )


def test_chunked_graph_size_is_flat_vs_batch_count():
    """Graph size must not grow with the number of batches.

    Asserted on autograd node counts; process memory is far too noisy to
    regress against. The forward pass runs under ``no_grad``, so
    the returned tensor carries only what this primitive contributes,
    regardless of batch count.
    """
    samples = _make_samples()
    theta = torch.tensor([0.7] * OUTPUT_SIZE, dtype=torch.float64,
                         requires_grad=True)

    node_counts = [
        count_autograd_nodes(
            chunked_vjp(
                lambda sl: _reduce(samples[sl], theta),
                _batches(N_SAMPLES, size),
                params=[theta],
            )
        )
        for size in _GRAPH_BATCH_SIZES
    ]

    assert len(set(node_counts)) == 1, (
        f"Graph size varied with batch count: {node_counts}"
    )

    naive_counts = [
        count_autograd_nodes(_accumulate_naively(samples, theta, size))
        for size in _GRAPH_BATCH_SIZES
    ]
    assert node_counts[-1] < naive_counts[-1], (
        f"At {N_SAMPLES // _GRAPH_BATCH_SIZES[-1]} batches the chunked graph "
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
    samples per batch would make the result depend on the batch size, the
    failure mode ``test_result_is_invariant_to_batch_size`` detects.
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
        _batches(num_rays, 16),
        params=[radius],
    )
    total.sum().backward()

    assert radius.grad is not None
    assert torch.allclose(radius.grad, radius_ref.grad, rtol=1e-10, atol=1e-12)


# --------------------------------------------------------------------------
# Guards.
#
# These cover the precondition the primitive cannot prevent, only detect: that
# batch_fn reproduces stage 1 when stage 3 re-evaluates it. Each case below
# returned a plausible, wrong gradient before the consistency check existed.
# --------------------------------------------------------------------------


def test_global_rng_batch_fn_is_made_reproducible():
    """Restoring the RNG turns a common mistake into a non-mistake.

    A batch_fn sampling through the global torch generator would otherwise
    draw different numbers in stage 3 from stage 1, differentiating a quantity
    that was never accumulated. Stage 1's generator state is restored before
    stage 3, so both passes draw the same numbers.

    This exact batch_fn raised the consistency error before RNG management was
    added, which is what the layer buys: not detection, but correctness.
    """
    samples = _make_samples()
    theta = torch.tensor([0.7] * OUTPUT_SIZE, dtype=torch.float64,
                         requires_grad=True)

    def sampling(sl):
        return torch.rand(OUTPUT_SIZE, dtype=torch.float64) * theta * samples[sl].sum()

    total = chunked_vjp(sampling, _batches(N_SAMPLES, 20), params=[theta])
    total.sum().backward()

    assert theta.grad is not None
    assert torch.all(torch.isfinite(theta.grad))
    assert torch.any(theta.grad != 0)


def test_backward_restores_the_callers_rng_state():
    """backward must leave the caller's random stream where it found it.

    Stage 3 rewinds the global generator to stage 1's state. Leaving it
    rewound would silently change every draw the caller makes afterwards, so
    the outer state is put back on exit.

    The batch_fn samples deliberately: with a deterministic one, backward
    consumes no randomness and the assertion would hold either way.
    """
    samples = _make_samples()
    theta = torch.tensor([0.7] * OUTPUT_SIZE, dtype=torch.float64,
                         requires_grad=True)

    def sampling(sl):
        return torch.rand(OUTPUT_SIZE, dtype=torch.float64) * theta * samples[sl].sum()

    total = chunked_vjp(sampling, _batches(N_SAMPLES, 20), params=[theta])

    # Draw between the passes, so the caller's stream is no longer where the
    # forward pass left it. Without this the restore is untestable: stage 3
    # replays stage 1's draws from stage 1's state and lands on the same
    # place regardless, so the assertion would hold either way.
    torch.rand(5)

    before = torch.get_rng_state()
    total.sum().backward()
    after = torch.get_rng_state()

    assert torch.equal(before, after)


def test_generator_advanced_across_calls_raises():
    """The case that restoring the global RNG would not fix.

    A generator created once outside batch_fn is advanced by stage 1, so stage
    3 continues from the advanced state. ``torch.set_rng_state`` resets the
    global stream, not a caller-held generator, so only the value comparison
    catches this.
    """
    samples = _make_samples()
    theta = torch.tensor([0.7] * OUTPUT_SIZE, dtype=torch.float64,
                         requires_grad=True)
    generator = torch.Generator().manual_seed(0)

    def drifting(sl):
        jitter = torch.rand(OUTPUT_SIZE, generator=generator, dtype=torch.float64)
        return jitter * theta * samples[sl].sum()

    with pytest.raises(RuntimeError, match="did not reproduce the forward pass"):
        total = chunked_vjp(drifting, _batches(N_SAMPLES, 20), params=[theta])
        total.sum().backward()


def test_iterator_batches_are_rejected_up_front():
    """Iterator batches raise before any tracing, not after.

    The forward pass would consume them and the backward pass would re-evaluate
    them as empty, returning a zero gradient. Rejecting them at the entry point
    costs nothing and names the fix, rather than reporting a mismatch after two
    full passes over the rays.
    """
    samples = _make_samples()
    theta = torch.tensor([0.7] * OUTPUT_SIZE, dtype=torch.float64,
                         requires_grad=True)
    consumed = [iter(range(0, 30)), iter(range(30, N_SAMPLES))]

    with pytest.raises(ValueError, match="are iterators"):
        chunked_vjp(
            lambda b: _reduce(samples[list(b)], theta), consumed, params=[theta]
        )


def test_consumable_non_iterator_batches_are_caught_by_the_check():
    """The backstop for consumables ``isinstance`` cannot see.

    An object may be re-iterable in form yet empty after the first pass, and
    such a thing is not an ``Iterator`` instance, so the entry check lets it
    through. The stage-1/stage-3 comparison catches it in backward: this is
    why both guards exist rather than only the cheap one.
    """

    class ConsumableOnce:
        def __init__(self, items):
            self._items = list(items)

        def __iter__(self):
            items, self._items = self._items, []
            return iter(items)

    samples = _make_samples()
    theta = torch.tensor([0.7] * OUTPUT_SIZE, dtype=torch.float64,
                         requires_grad=True)
    once = [ConsumableOnce(range(0, 30)), ConsumableOnce(range(30, N_SAMPLES))]

    with pytest.raises(RuntimeError, match="did not reproduce the forward pass"):
        total = chunked_vjp(
            lambda b: _reduce(samples[list(b)], theta), once, params=[theta]
        )
        total.sum().backward()


def test_non_finite_total_does_not_trip_the_check():
    """Vignetted rays contribute NaN, and that is normal, not a violation.

    The tolerance is derived from the finite entries alone. Taking ``max()``
    over the raw total would return NaN, making every comparison fail and
    rejecting every trace with a blocked ray in it.
    """
    samples = _make_samples()
    samples[3] = float("nan")
    theta = torch.tensor([0.7] * OUTPUT_SIZE, dtype=torch.float64,
                         requires_grad=True)

    total = chunked_vjp(
        lambda sl: (samples[sl] * theta).sum(dim=0),
        _batches(N_SAMPLES, 20),
        params=[theta],
    )
    total.nan_to_num(0.0).sum().backward()

    assert theta.grad is not None


def test_tolerance_scales_with_dtype():
    """float32 round-off must not be mistaken for non-determinism.

    A tolerance sized for float64 would reject every float32 reduction.
    """
    with backend_state("torch", precision="float32"):
        samples = torch.randn(
            N_SAMPLES, OUTPUT_SIZE, generator=torch.Generator().manual_seed(0)
        )
        theta = torch.tensor([0.7] * OUTPUT_SIZE, requires_grad=True)

        total = chunked_vjp(
            lambda sl: torch.tanh(samples[sl] * theta).sum(dim=0),
            _batches(N_SAMPLES, 7),
            params=[theta],
        )
        total.sum().backward()

    assert theta.grad is not None
    assert torch.all(torch.isfinite(theta.grad))
