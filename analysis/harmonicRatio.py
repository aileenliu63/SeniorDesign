import numpy as np
from scipy.signal import find_peaks

def harmonic_ratio(signal, fs, num_harmonics=10):
    """
    Computes Harmonic Ratio (HR) for a 1D gyro/accel gait signal.

    Parameters
    ----------
    signal : list or np.array
        1D array of angular velocity (e.g., gz)
    fs : float
        Sampling rate in Hz
    num_harmonics : int
        Number of harmonics to include (default = 10)

    Returns
    -------
    HR : float
        Harmonic Ratio (even harmonic energy / odd harmonic energy)
    """
    signal = np.array(signal)

    # Remove mean to improve FFT quality
    signal_d = signal - np.mean(signal)

    # FFT
    N = len(signal_d)
    freqs = np.fft.rfftfreq(N, 1/fs)
    fft_vals = np.abs(np.fft.rfft(signal_d))

    # ---- Estimate step frequency using peak in gyro magnitude ----
    # Step frequency usually ~0.8–2.5 Hz in adults
    valid = (freqs > 0.5) & (freqs < 3.5)
    dom_idx = np.argmax(fft_vals[valid])
    step_freq = freqs[valid][dom_idx]

    # ---- Collect harmonic amplitudes ----
    harmonics_even = []
    harmonics_odd = []

    for h in range(1, num_harmonics+1):
        target_freq = h * step_freq
        idx = np.argmin(np.abs(freqs - target_freq))

        if h % 2 == 0:
            harmonics_even.append(fft_vals[idx])
        else:
            harmonics_odd.append(fft_vals[idx])

    # Avoid division by zero
    if sum(harmonics_odd) == 0:
        return np.nan

    HR = sum(harmonics_even) / sum(harmonics_odd)
    return HR
