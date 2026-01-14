import numpy as np
import pytest

from wavelet_excitation import WaveletSignalGenerator


# --- Fixtures ---
@pytest.fixture
def default_gen():
    """Returns a standard generator instance."""
    return WaveletSignalGenerator(
        db_order=10, dt=1e-4, t_sim=1.0, cutoff_freq=500, target_pdf_type="uniform", random_seed=42
    )


# --- Initialization Tests ---
def test_initialization_valid():
    gen = WaveletSignalGenerator(dt=1e-3, t_sim=1.0, cutoff_freq=100)
    assert gen.num_points == 1000
    assert gen.nyquist == 500.0


def test_initialization_invalid_time():
    with pytest.raises(ValueError, match="positive"):
        WaveletSignalGenerator(dt=-1e-3, t_sim=1.0)


def test_initialization_invalid_cutoff():
    dt = 1e-3
    nyquist = 0.5 / dt  # 500 Hz
    with pytest.raises(ValueError, match="Nyquist"):
        WaveletSignalGenerator(dt=dt, t_sim=1.0, cutoff_freq=600)


# --- Generation Tests ---
def test_generation_shape_single(default_gen):
    sig = default_gen.generate()
    assert sig.ndim == 1
    assert len(sig) == default_gen.num_points


def test_generation_shape_mimo(default_gen):
    num_channels = 3
    sig = default_gen.generate(num_channels=num_channels)
    assert sig.ndim == 2
    assert sig.shape == (num_channels, default_gen.num_points)


def test_signal_scaling(default_gen):
    limits = (-2.0, 2.0)
    sig = default_gen.generate(scale_limits=limits)
    assert np.isclose(np.min(sig), limits[0])
    assert np.isclose(np.max(sig), limits[1])


def test_reproducibility():
    """Ensures that the same seed produces the exact same signal."""
    # Run 1
    gen1 = WaveletSignalGenerator(dt=1e-4, t_sim=1.0, cutoff_freq=500, random_seed=42)
    sig1 = gen1.generate(num_channels=1, seeds=123)

    # Run 2 (New instance with same parameters)
    gen2 = WaveletSignalGenerator(dt=1e-4, t_sim=1.0, cutoff_freq=500, random_seed=42)
    sig2 = gen2.generate(num_channels=1, seeds=123)

    np.testing.assert_array_equal(sig1, sig2)


def test_mimo_independence(default_gen):
    """Ensures that MIMO channels are not identical."""
    sig = default_gen.generate(num_channels=2, seeds=100)
    # Check that channel 0 is not equal to channel 1
    assert not np.array_equal(sig[0], sig[1])

    # Check correlation is low (sanity check)
    corr = np.corrcoef(sig[0], sig[1])[0, 1]
    assert abs(corr) < 0.3  # Loose bound for unit test, depends on length


def test_pre_filter_warning():
    """Check if warning is issued when filter cutoff is too high."""
    with pytest.warns(UserWarning, match="Filter cutoff.*>= Nyquist"):
        # Create a generator where filter cutoff exceeds Nyquist
        gen = WaveletSignalGenerator(dt=1e-4, t_sim=1.0, cutoff_freq=1000)  # Nyquist=5000
        gen.generate(pre_filter=True, filter_cutoff=6000)  # Exceeds Nyquist
