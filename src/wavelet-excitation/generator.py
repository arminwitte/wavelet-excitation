import numpy as np
import pywt
import warnings
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from typing import Optional, Tuple, Literal, List, Union

class WaveletSignalGenerator:
    """
    A signal generator that reproduces 'Wavelet-based Excitation Signals' for system identification.
    
    This class implements the 'Non-Gaussian Simulation' algorithm by Masters & Gurley (2003).
    It generates a signal that combines:
    1. The Spectral Envelope of a Daubechies scaling function (Broadband/Low-pass).
    2. A Target Probability Density Function (PDF), typically Uniform (Top-Hat) for high crest factors.

    Original methodology adapted from:
    Föller, S., & Polifke, W. (2011). Advances in identification techniques...
    """

    # --- Configuration Constants ---
    DEFAULT_WAVELET_LEVEL = 10
    FALLBACK_WAVELET_LEVEL = 8
    
    # 0.707 (-3dB) in Amplitude domain corresponds to 0.5 (-3dB) in Power domain
    DB_3DB_THRESHOLD_AMPLITUDE = 1.0 / np.sqrt(2.0)
    
    DEFAULT_PREFILTER_FACTOR = 1.2
    MAX_NYQUIST_FRACTION = 0.95
    
    PLOT_BINS = 80

    def __init__(self, 
                 db_order: int = 20, 
                 dt: float = 1e-5, 
                 t_sim: float = 0.1, 
                 cutoff_freq: Optional[float] = None, 
                 target_pdf_type: Literal['uniform', 'wavelet'] = 'uniform',
                 random_seed: Optional[int] = None) -> None:
        """
        Initialize the generator.

        Args:
            db_order: Order of the Daubechies wavelet (e.g., 20 for 'db20').
            dt: Time step in seconds [s].
            t_sim: Total simulation duration in seconds [s].
            cutoff_freq: Desired 3dB cut-off frequency in Hz. Must be < Nyquist.
            target_pdf_type: 'uniform' (Top-Hat, default) or 'wavelet' (Raw Fractal).
            random_seed: Base seed for reproducibility. 
                         If num_channels > 1 in generate(), this seed is incremented.
        """
        # 1. Input Validation
        if t_sim <= 0 or dt <= 0:
            raise ValueError("Time parameters (t_sim, dt) must be positive.")
        
        self.nyquist = 0.5 / dt
        if cutoff_freq is not None and cutoff_freq >= self.nyquist:
            raise ValueError(f"cutoff_freq ({cutoff_freq} Hz) must be < Nyquist ({self.nyquist} Hz).")

        self.db_order = db_order
        self.dt = dt
        self.t_sim = t_sim
        self.cutoff_freq = cutoff_freq
        self.target_pdf_type = target_pdf_type
        self.seed = random_seed
        
        # 2. Setup Time Axis
        self.num_points = int(t_sim / dt)
        self.time_axis = np.arange(self.num_points) * dt
        
        # Internal storage
        self.target_amplitude_spectrum: Optional[np.ndarray] = None
        self.target_distribution: Optional[np.ndarray] = None
        # generated_signal can be 1D array or 2D array (channels, time)
        self.generated_signal: Optional[np.ndarray] = None

        # Generate targets
        self._generate_target_statistics()

    def _generate_target_statistics(self) -> None:
        """Generates Target Spectrum and Target CDF."""
        # 1. Generate high-res scaling function (phi)
        wavelet = pywt.Wavelet(f'db{self.db_order}')
        try:
            phi, _, _ = wavelet.wavefun(level=self.DEFAULT_WAVELET_LEVEL)
        except AttributeError:
            phi, _, _ = wavelet.wavefun(level=self.FALLBACK_WAVELET_LEVEL)

        # 2. Rescale bandwidth (Time-Scaling of Spectrum)
        if self.cutoff_freq is not None:
            phi = self._rescale_wavelet_to_bandwidth(phi, self.cutoff_freq)

        # 3. Target Distribution (PDF/CDF)
        if self.target_pdf_type == 'uniform':
            # Top-Hat / Rectangle PDF -> Linear CDF
            self.target_distribution = np.linspace(-1, 1, self.num_points)
        else:
            # 'wavelet': Use the raw fractal distribution of the scaling function
            phi_sorted = np.sort(phi)
            target_indices = np.linspace(0, 1, len(phi_sorted))
            signal_indices = np.linspace(0, 1, self.num_points)
            self.target_distribution = np.interp(signal_indices, target_indices, phi_sorted)

        # 4. Target Spectrum (Spectral Envelope)
        if len(phi) > self.num_points:
            warnings.warn(
                f"Wavelet length ({len(phi)}) exceeds simulation points ({self.num_points}). "
                "Target spectrum will be truncated.", UserWarning
            )
            phi_padded = phi[:self.num_points]
        else:
            phi_padded = np.zeros(self.num_points)
            phi_padded[:len(phi)] = phi

        # Calculate Amplitude Spectrum
        self.target_amplitude_spectrum = np.abs(fft(phi_padded))

    def _rescale_wavelet_to_bandwidth(self, phi_base: np.ndarray, target_f3db: float) -> np.ndarray:
        """Resamples the wavelet vector to shift its spectral 3dB point."""
        n_base = len(phi_base)
        freqs = fftfreq(n_base, self.dt)
        spec = np.abs(fft(phi_base))
        
        pos_mask = freqs >= 0
        f_pos = freqs[pos_mask]
        s_pos = spec[pos_mask]
        
        if s_pos[0] == 0: s_pos[0] = 1e-10
        s_norm = s_pos / s_pos[0]
        
        # Robust 3dB detection
        crossings = np.where(np.diff(np.sign(s_norm - self.DB_3DB_THRESHOLD_AMPLITUDE)))[0]
        
        if len(crossings) == 0:
            current_f3db = f_pos[-1] 
        else:
            idx = crossings[0]
            y1, y2 = s_norm[idx], s_norm[idx+1]
            x1, x2 = f_pos[idx], f_pos[idx+1]
            current_f3db = x1 + (self.DB_3DB_THRESHOLD_AMPLITUDE - y1) * (x2 - x1) / (y2 - y1)

        if current_f3db <= 1e-6: 
            return phi_base

        scale_factor = current_f3db / target_f3db
        new_len = int(n_base * scale_factor)
        
        if new_len < 2: return phi_base
            
        old_indices = np.linspace(0, 1, n_base)
        new_indices = np.linspace(0, 1, new_len)
        
        # Cubic interp with zero-fill (compact support)
        interpolator = interp1d(old_indices, phi_base, kind='cubic', 
                                fill_value=0.0, bounds_error=False)
        return interpolator(new_indices)

    def _apply_spectral_constraint(self, signal: np.ndarray) -> np.ndarray:
        """Enforces Target Spectrum preserving Phase and Energy."""
        sig_fft = fft(signal)
        sig_phase = np.angle(sig_fft)
        
        current_energy = np.sum(np.abs(sig_fft)**2)
        target_energy_shape = np.sum(self.target_amplitude_spectrum**2)
        
        if target_energy_shape > 0:
            scaling_factor = np.sqrt(current_energy / target_energy_shape)
            scaled_target = self.target_amplitude_spectrum * scaling_factor
        else:
            scaled_target = self.target_amplitude_spectrum

        new_fft = scaled_target * np.exp(1j * sig_phase)
        return np.real(ifft(new_fft))

    def _apply_pdf_constraint(self, signal: np.ndarray) -> np.ndarray:
        """Enforces Target CDF preserving Rank Ordering."""
        rank_indices = np.argsort(signal)
        corrected_signal = np.zeros_like(signal)
        
        # Map directly 1-to-1 via ranks using pre-calculated target distribution
        corrected_signal[rank_indices] = self.target_distribution
        
        return corrected_signal

    def _generate_single_core(self, 
                              max_iter: int, 
                              tol: float) -> np.ndarray:
        """Core logic for generating one signal iteration."""
        current_signal = np.random.randn(self.num_points)
        prev_signal = None
        
        for i in range(max_iter):
            current_signal = self._apply_spectral_constraint(current_signal)
            current_signal = self._apply_pdf_constraint(current_signal)
            
            if prev_signal is not None:
                diff = np.linalg.norm(current_signal - prev_signal)
                norm = np.linalg.norm(current_signal)
                if norm > 0 and (diff / norm) < tol:
                    break 
            
            prev_signal = current_signal.copy()
            
        return current_signal

    def generate(self, 
                 max_iter: int = 50, 
                 tol: float = 1e-4, 
                 pre_filter: bool = False, 
                 filter_cutoff: Optional[float] = None,
                 scale_limits: Optional[Tuple[float, float]] = (-1, 1),
                 num_channels: int = 1,
                 seeds: Optional[Union[int, List[int]]] = None) -> np.ndarray:
        """
        Runs the iterative simulation. Can generate multiple uncorrelated signals (MIMO).
        
        Args:
            max_iter: Maximum iterations per signal.
            tol: Convergence tolerance.
            pre_filter: Apply Butterworth low-pass filter at end.
            filter_cutoff: Manual cutoff for pre-filter.
            scale_limits: Final scaling range (min, max).
            num_channels: Number of independent signals to generate (Default: 1).
            seeds: Control for random seeds.
                   - If int: Used as start value, incremented for each channel (e.g. 100 -> 100, 101, ...).
                   - If List[int]: Specific seeds for each channel (len must match num_channels).
                   - If None: Uses self.seed (incremented) if set, otherwise random.
        
        Returns:
            np.ndarray: 
                - If num_channels == 1: 1D Array (num_points,)
                - If num_channels > 1:  2D Array (num_channels, num_points)
        """
        # 1. Resolve Seeds
        seed_list = []
        if seeds is not None:
            if isinstance(seeds, int):
                seed_list = [seeds + i for i in range(num_channels)]
            elif isinstance(seeds, list):
                if len(seeds) != num_channels:
                    raise ValueError(f"Length of seed list ({len(seeds)}) must match num_channels ({num_channels}).")
                seed_list = seeds
            else:
                raise TypeError("seeds must be int or List[int]")
        elif self.seed is not None:
            seed_list = [self.seed + i for i in range(num_channels)]
        else:
            seed_list = [None] * num_channels # No enforced seeds

        # 2. Generation Loop
        signals = []
        for i in range(num_channels):
            # Set seed for this specific channel
            if seed_list[i] is not None:
                np.random.seed(seed_list[i])
            
            # Run core logic
            raw_sig = self._generate_single_core(max_iter, tol)
            
            # Post-processing (Filtering)
            if pre_filter:
                raw_sig = self._apply_post_filter_to_signal(raw_sig, filter_cutoff)
                
            # Post-processing (Scaling)
            if scale_limits is not None:
                raw_sig = self._scale_signal_array(raw_sig, scale_limits)
                
            signals.append(raw_sig)

        # 3. Store and Return
        if num_channels == 1:
            self.generated_signal = signals[0]
        else:
            self.generated_signal = np.vstack(signals)
            
        return self.generated_signal

    def _apply_post_filter_to_signal(self, signal: np.ndarray, user_cutoff: Optional[float]) -> np.ndarray:
        """Helper to apply filter to a specific array."""
        if user_cutoff is not None:
            fc = user_cutoff
        elif self.cutoff_freq:
            fc = min(self.DEFAULT_PREFILTER_FACTOR * self.cutoff_freq, 
                     self.MAX_NYQUIST_FRACTION * self.nyquist)
        else:
            fc = 0.8 * self.nyquist

        if fc >= self.nyquist:
            warnings.warn(f"Filter cutoff {fc:.1f} >= Nyquist. Filtering skipped.")
            return signal

        b, a = butter(5, fc / self.nyquist, btype='low')
        return filtfilt(b, a, signal)

    def _scale_signal_array(self, signal: np.ndarray, limits: Tuple[float, float]) -> np.ndarray:
        """Helper to scale a specific array."""
        s_min, s_max = np.min(signal), np.max(signal)
        if s_max > s_min:
            d_min, d_max = limits
            norm = (signal - s_min) / (s_max - s_min)
            return norm * (d_max - d_min) + d_min
        return signal

    def plot_diagnostics(self) -> None:
        """Visualizes Time Series, PDF, and Magnitude Spectrum. Handles MIMO (plots channel 0)."""
        if self.generated_signal is None:
            print("No signal generated. Run .generate() first.")
            return

        # Handle MIMO shape
        if self.generated_signal.ndim > 1:
            signal_to_plot = self.generated_signal[0]
            mimo_note = f" (Channel 0 of {self.generated_signal.shape[0]})"
        else:
            signal_to_plot = self.generated_signal
            mimo_note = ""

        fig, axs = plt.subplots(3, 1, figsize=(10, 12), constrained_layout=True)
        
        # 1. Time Series
        title_str = f"Generated Signal{mimo_note} (db{self.db_order}, PDF={self.target_pdf_type}"
        if self.cutoff_freq:
            title_str += f", fc={self.cutoff_freq}Hz"
        title_str += ")"
        
        axs[0].plot(self.time_axis, signal_to_plot, color='#1f77b4', lw=0.8)
        axs[0].set_title(title_str)
        axs[0].set_xlabel("Time [s]")
        axs[0].set_ylabel("Amplitude")
        axs[0].grid(True, alpha=0.3)
        axs[0].set_xlim(0, self.t_sim)

        # 2. PDF
        axs[1].hist(signal_to_plot, bins=self.PLOT_BINS, density=True, alpha=0.6, 
                    color='#1f77b4', label='Generated')
        
        s_min, s_max = signal_to_plot.min(), signal_to_plot.max()
        t_norm = self.target_distribution
        if t_norm.max() > t_norm.min():
             t_scaled = (t_norm - t_norm.min())/(t_norm.max() - t_norm.min()) * (s_max - s_min) + s_min
        else:
             t_scaled = t_norm
             
        axs[1].hist(t_scaled, bins=self.PLOT_BINS, density=True, alpha=0.4, 
                    color='red', histtype='step', lw=2, label=f'Target ({self.target_pdf_type})')
        axs[1].set_title("Probability Density Function (PDF)")
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)

        # 3. Magnitude Spectrum
        freqs = fftfreq(self.num_points, self.dt)
        mask = freqs > 0
        f_pos = freqs[mask]
        
        gen_fft = np.abs(fft(signal_to_plot))[mask]
        gen_fft_norm = gen_fft / np.max(gen_fft)
        
        target_fft = self.target_amplitude_spectrum[mask]
        target_fft_norm = target_fft / np.max(target_fft)

        axs[2].loglog(f_pos, gen_fft_norm, label='Generated', color='#1f77b4', alpha=0.9)
        axs[2].loglog(f_pos, target_fft_norm, label='Target Envelope', 
                      color='red', ls='--', lw=1.5, alpha=0.9)
        
        if self.cutoff_freq:
            axs[2].axvline(self.cutoff_freq, color='green', ls=':', label=f'Cutoff {self.cutoff_freq} Hz')

        axs[2].set_title("Magnitude Spectrum (Normalized)")
        axs[2].set_xlabel("Frequency [Hz]")
        axs[2].legend()
        axs[2].grid(True, which="both", alpha=0.3)
        axs[2].set_xlim(f_pos[0], f_pos[-1])

        plt.show()

# --- Validator ---

def validate_mimo_independence(signals: Union[List[np.ndarray], np.ndarray]):
    """Calculates and prints the Cross-Correlation Matrix of the signals."""
    # Convert list to array or handle array directly
    if isinstance(signals, list):
        sigs_array = np.vstack(signals)
    else:
        sigs_array = signals
        
    if sigs_array.ndim == 1:
        print("Single signal provided. Independence check irrelevant.")
        return

    num = sigs_array.shape[0]
    corr_matrix = np.zeros((num, num))
    
    print("\n--- MIMO Independence Check (Correlation Coefficient) ---")
    for i in range(num):
        for j in range(num):
            corr = np.corrcoef(sigs_array[i], sigs_array[j])[0, 1]
            corr_matrix[i, j] = corr
    
    print(np.round(corr_matrix, 4))
    
    off_diag = corr_matrix[~np.eye(num, dtype=bool)]
    max_corr = np.max(np.abs(off_diag)) if len(off_diag) > 0 else 0.0
    print(f"Max absolute cross-correlation: {max_corr:.4f}")
    if max_corr < 0.05:
        print(">> PASS: Signals are effectively uncorrelated.")
    else:
        print(">> WARNING: Significant correlation detected (Expected for short durations).")

# --- Smoke Test & MIMO Demo ---
if __name__ == "__main__":
    try:
        # 1. Single Signal Test
        print("1. Testing Single Signal Generation...")
        gen = WaveletSignalGenerator(
            db_order=20, 
            dt=1e-5, 
            t_sim=0.1, 
            cutoff_freq=1500,  
            target_pdf_type='uniform', 
            random_seed=42
        )
        # Old API works (num_channels default is 1)
        sig = gen.generate(max_iter=50, tol=1e-4, pre_filter=False, scale_limits=(-1, 1))
        
        # 2. MIMO Test (Integrated)
        print("\n2. Testing MIMO Generation (3 Channels) inside generate()...")
        mimo_sigs = gen.generate(
            num_channels=3,
            seeds=100, # Will produce seeds 100, 101, 102
            scale_limits=(-1, 1),
            pre_filter=False
        )
        
        print(f"Output Shape: {mimo_sigs.shape}")
        validate_mimo_independence(mimo_sigs)
        
    except Exception as e:
        print(f"Error: {e}")