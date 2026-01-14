# Wavelet-Based Excitation Signal Generator

A Python implementation of the **Non-Gaussian Simulation algorithm** by Masters & Gurley (2003), adapted for system identification using **Daubechies Wavelet** scaling functions.

Designed for Data Scientists and Engineers who need **high-energy excitation signals** with **strict bandwidth limits**.

## 🚀 The Problem it Solves

In System Identification (e.g., CFD, Robotics, Control Systems), excitation signals often face conflicting requirements:

1. **Bandwidth:** You need to excite a specific frequency range (Broadband/Low-pass).

2. **Constraints:** Actuators have hard physical limits (e.g., valve position 0-100%, max torque).

3. **Energy:** You want to inject maximum energy to overcome noise (high SNR).

**Standard approach:** Filtering a binary signal (PRBS) or white noise creates a **Gaussian distribution**. To stay within limits ($\pm 1$), a Gaussian signal must have a low RMS energy (High Crest Factor \~3-4), wasting potential excitation power.

**This Solution:** This generator iteratively shapes the signal to have a **block-like (Uniform) amplitude distribution** while maintaining the spectral shape of a wavelet.

## ✨ Features

* **Bandwidth Control:** Precise 3dB cutoff via Daubechies scaling functions.

* **Energy Maximization:** Uniform (Top-Hat) PDF target for minimal Crest Factor (\~1.73).

* **MIMO Ready:** Easy generation of uncorrelated multi-channel signals.

* **Modern Stack:** Built with `uv`, tested with `pytest`, formatted with `ruff`.

## 📦 Installation

This project uses modern Python tooling.

### Using `uv` (Recommended)

```bash
# Install directly from GitHub
uv pip install git+https://github.com/YOUR_USERNAME/wavelet-excitation.git
```

### For development:

```bash
git clone https://github.com/YOUR_USERNAME/wavelet-excitation.git
cd wavelet-excitation

# Create venv and install dependencies + dev tools
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Using standard pip

```bash
pip install wavelet-excitation
```

(Once published to PyPI)

## ⚡ Usage

```python
from wavelet_excitation import WaveletSignalGenerator

# Initialize (Uniform PDF = Max Energy)
# target_pdf_type='uniform' (Default) creates an ideal Top-Hat distribution.
gen = WaveletSignalGenerator(dt=1e-4, t_sim=10.0, cutoff_freq=1500, target_pdf_type='uniform')

# Generate MIMO signals (2 Channels)
# Returns a (2, N) array of uncorrelated signals
signals = gen.generate(num_channels=2, seeds=42, scale_limits=(-1, 1))

gen.plot_diagnostics()
```

## 📝 Methodological Note: Uniform vs. Maximum Entropy

This implementation offers an improvement over the original MATLAB reference (Föller & Polifke, 2011):

**Original Method (MATLAB):** Used a constrained Maximum Entropy Method (MEM) solver to find a PDF that matched the first $N$ moments of a Daubechies wavelet. This resulted in a "wavy" Top-Hat distribution due to polynomial approximation.

**This Implementation (Python):** Defaults to target_pdf_type='uniform', which analytically represents the limit of infinite moments (a perfect Top-Hat).

**Benefit 1:** Slightly higher energy efficiency (lower Crest Factor).  
**Benefit 2:** Guaranteed zero-mean symmetry.  
**Benefit 3:** Faster execution (no iterative MEM solver needed).

## 📚 References & Acknowledgements

This Python package is a modernization and port of original MATLAB implementations developed at the Technical University of Munich (TUM).

**Primary References:**  
1. Masters, F., & Gurley, K. (2003). Non-Gaussian Simulation: Cumulative Distribution Function Map-Based Spectral Correction. Journal of Engineering Mechanics.  
2. Föller, S., & Polifke, W. (2011). Advances in Identification Techniques for Aero-Acoustic Scattering Coefficients. ICSV18.

**Acknowledgements:**  
Special thanks to Dr. Stephan Föller for the original research and the MATLAB reference implementation (SFSI_DaWaGe, SFSI_TSerGen) on which this Python port is based.

## 🖊️ Citing

If you use this software in your research, please cite it as follows:

```bibtex
@software{wavelet_excitation_2026,
  author = {Witte, Armin},
  title = {Wavelet-Based Excitation Signal Generator},
  year = {2026},
  url = {https://github.com/YOUR_USERNAME/wavelet-excitation},
  version = {0.2.0}
}
```

## 📄 License

MIT License