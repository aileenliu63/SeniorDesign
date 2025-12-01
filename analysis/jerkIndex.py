# analysis/jerk.py

import numpy as np

def jerk_rms(signal, fs):
    """
    Compute angular jerk RMS from a 1D gyro signal.

    Parameters
    ----------
    signal : list or np.array
        Angular velocity time series (e.g., gz)
    fs : float
        Sampling frequency in Hz

    Returns
    -------
    float
        RMS of angular jerk (dω/dt), same units as signal per second.
    """
    sig = np.array(signal, dtype=float)

    if len(sig) < 2:
        return np.nan

    # Finite difference derivative: dω/dt ≈ Δω * fs
    jerk = np.diff(sig) * fs

    # RMS of jerk
    return np.sqrt(np.mean(jerk**2))
