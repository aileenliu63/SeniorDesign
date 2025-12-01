# analysis/step_plots.py

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 0. Step detection (gyro)
# ============================================================
def detect_steps_from_gyro(
    signal,
    fs,
    min_distance_s=0.4,
    threshold_factor=0.2,
):
    """
    Very simple peak-based step detector on 1D gyro signal.

    Parameters
    ----------
    signal : array-like
        1D gyro signal (e.g., gy or gz), shape (N,)
    fs : float
        Sampling frequency (Hz)
    min_distance_s : float
        Minimum time between steps in seconds (refractory period)
    threshold_factor : float
        Fraction of peak-to-peak range to use as threshold.
        threshold = min(signal) + threshold_factor * (max - min)

    Returns
    -------
    step_indices : np.ndarray
        Indices (sample numbers) of detected steps.
    """
    sig = np.asarray(signal, dtype=float)
    n = len(sig)
    if n < 3:
        return np.array([], dtype=int)

    # Basic detrend
    sig = sig - np.mean(sig)

    # Amplitude threshold
    s_min, s_max = np.min(sig), np.max(sig)
    thr = s_min + threshold_factor * (s_max - s_min)

    min_dist_samples = int(min_distance_s * fs)
    step_indices = []

    last_idx = -min_dist_samples - 1

    for i in range(1, n - 1):
        # local maximum
        if sig[i] > sig[i - 1] and sig[i] > sig[i + 1] and sig[i] > thr:
            if i - last_idx >= min_dist_samples:
                step_indices.append(i)
                last_idx = i

    return np.array(step_indices, dtype=int)


# ============================================================
# 0b. Step detection (accel)
# ============================================================
def detect_steps_from_accel(
    signal,
    fs,
    min_distance_s=0.4,
    threshold_factor=0.15,
):
    """
    Simple peak-based step detector on 1D accelerometer signal
    (e.g., vertical acceleration).

    Parameters
    ----------
    signal : array-like
        1D accel signal (e.g., az), shape (N,)
    fs : float
        Sampling frequency (Hz)
    min_distance_s : float
        Minimum time between steps in seconds (refractory period)
    threshold_factor : float
        Fraction of peak-to-peak range to use as threshold.

    Returns
    -------
    step_indices : np.ndarray
        Indices (sample numbers) of detected steps.
    """
    sig = np.asarray(signal, dtype=float)
    n = len(sig)
    if n < 3:
        return np.array([], dtype=int)

    sig = sig - np.mean(sig)

    s_min, s_max = np.min(sig), np.max(sig)
    thr = s_min + threshold_factor * (s_max - s_min)

    min_dist_samples = int(min_distance_s * fs)
    step_indices = []
    last_idx = -min_dist_samples - 1

    for i in range(1, n - 1):
        if sig[i] > sig[i - 1] and sig[i] > sig[i + 1] and sig[i] > thr:
            if i - last_idx >= min_dist_samples:
                step_indices.append(i)
                last_idx = i

    return np.array(step_indices, dtype=int)


# ============================================================
# Helper: derive step/stride metrics (generic)
# ============================================================
def compute_step_metrics(step_indices, fs, signal=None):
    """
    Compute basic temporal and amplitude metrics from step indices.

    Parameters
    ----------
    step_indices : array-like
        Sample indices of detected steps.
    fs : float
        Sampling frequency (Hz).
    signal : array-like or None
        Optional 1D signal. If provided, peak value per step
        will be taken as |signal[step_index]|.

    Returns
    -------
    metrics : dict with keys
        'step_times' : np.ndarray, shape (N_steps-1,)
        'stride_times' : np.ndarray, shape (≈N_steps/2 - 1,)
        'peak_value' : np.ndarray, shape (N_steps,) or None
    """
    step_indices = np.asarray(step_indices, dtype=int)
    metrics = {
        "step_times": np.array([]),
        "stride_times": np.array([]),
        "peak_value": None,
    }

    if step_indices.size < 2:
        return metrics

    # Step times (time between consecutive steps)
    metrics["step_times"] = np.diff(step_indices) / fs

    # Stride times: approximate as time between every other step
    # (same "foot" if you assume alternating steps)
    if step_indices.size >= 3:
        stride_indices = step_indices[::2]  # even steps
        if stride_indices.size >= 2:
            metrics["stride_times"] = np.diff(stride_indices) / fs

    # Peak value per step (useful for gyro or accel)
    if signal is not None:
        sig = np.asarray(signal, dtype=float)
        metrics["peak_value"] = np.abs(sig[step_indices])

    return metrics


# ============================================================
# (A) Raw signal with step markers (generic)
# ============================================================
def plot_raw_with_steps(
    signal,
    fs,
    step_indices,
    title="(A) Raw signal with step markers",
    t=None,
    ylabel="Signal",
):
    """
    (A) Raw signal with step markers.

    Parameters
    ----------
    signal : array-like
        1D signal (gyro or accel), shape (N,)
    fs : float
        Sampling frequency (Hz)
    step_indices : array-like
        Detected step indices
    title : str
        Plot title
    t : array-like or None
        Optional time axis. If None, t = np.arange(N)/fs.
    ylabel : str
        Y axis label (e.g. 'Angular velocity' or 'Acceleration')
    """
    sig = np.asarray(signal, dtype=float)
    n = len(sig)

    if t is None:
        t = np.arange(n) / fs
    else:
        t = np.asarray(t)

    step_indices = np.asarray(step_indices, dtype=int)

    plt.figure(figsize=(10, 4))
    plt.plot(t, sig, label="signal")

    # Color even/odd steps differently (approx left/right)
    for k, idx in enumerate(step_indices):
        color = "tab:blue" if (k % 2 == 0) else "tab:orange"
        if 0 <= idx < n:
            plt.axvline(
                t[idx],
                color=color,
                alpha=0.7,
                linestyle="--",
                linewidth=1.0,
            )

    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(["signal", "step markers (even/odd)"])
    plt.tight_layout()


def plot_raw_gyro_with_steps(
    signal,
    fs,
    step_indices,
    title="(A) Raw angular velocity with step markers",
    t=None,
):
    """Wrapper for gyro-specific labeling."""
    plot_raw_with_steps(
        signal=signal,
        fs=fs,
        step_indices=step_indices,
        title=title,
        t=t,
        ylabel="Angular velocity",
    )


def plot_raw_accel_with_steps(
    signal,
    fs,
    step_indices,
    title="(A) Raw acceleration with step markers",
    t=None,
):
    """Wrapper for accel-specific labeling."""
    plot_raw_with_steps(
        signal=signal,
        fs=fs,
        step_indices=step_indices,
        title=title,
        t=t,
        ylabel="Acceleration",
    )


# ============================================================
# (B) Step time series plot
# ============================================================
def plot_step_time_series(
    step_times,
    title="(B) Step time series plot",
):
    """
    (B) Step time series plot.

    Parameters
    ----------
    step_times : array-like
        Step durations in seconds (between consecutive steps).
    title : str
        Plot title.
    """
    st = np.asarray(step_times, dtype=float)
    if st.size == 0:
        print("No step times to plot.")
        return

    idx = np.arange(1, len(st) + 1)  # step index (1..N)
    mean_st = np.mean(st)
    std_st = np.std(st)

    plt.figure(figsize=(8, 4))
    plt.plot(idx, st, marker="o", linestyle="-", label="Step time")

    # Mean line
    plt.axhline(mean_st, color="tab:red", linestyle="--", label="Mean")

    # Shaded band: mean ± 1 SD
    plt.fill_between(
        idx,
        mean_st - std_st,
        mean_st + std_st,
        color="tab:red",
        alpha=0.2,
        label="Mean ± 1 SD",
    )

    plt.xlabel("Step index")
    plt.ylabel("Step time (s)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()


# ============================================================
# (C) Stride time series plot
# ============================================================
def plot_stride_time_series(
    stride_times,
    title="(C) Stride time series plot",
):
    """
    (C) Stride time series plot.

    One point per stride: approximate as time between
    every other step (same foot to same foot).

    Parameters
    ----------
    stride_times : array-like
        Stride durations in seconds.
    title : str
        Plot title.
    """
    st = np.asarray(stride_times, dtype=float)
    if st.size == 0:
        print("No stride times to plot.")
        return

    idx = np.arange(1, len(st) + 1)  # stride index
    mean_st = np.mean(st)
    std_st = np.std(st)

    plt.figure(figsize=(8, 4))
    plt.plot(idx, st, marker="o", linestyle="-", label="Stride time")

    plt.axhline(mean_st, color="tab:green", linestyle="--", label="Mean")
    plt.fill_between(
        idx,
        mean_st - std_st,
        mean_st + std_st,
        color="tab:green",
        alpha=0.2,
        label="Mean ± 1 SD",
    )

    plt.xlabel("Stride index")
    plt.ylabel("Stride time (s)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()


# ============================================================
# (D) Peak value per step (gyro or accel)
# ============================================================
def plot_peak_value_per_step(
    peak_values,
    title="(D) Peak value per step",
    ylabel="Peak value",
):
    """
    (D) Peak value per step.

    Parameters
    ----------
    peak_values : array-like
        Peak signal values per step (e.g. |gyro[step_index]| or |accel[step_index]|.
    title : str
        Plot title.
    ylabel : str
        Y axis label (e.g. 'Peak angular velocity' or 'Peak acceleration')
    """
    po = np.asarray(peak_values, dtype=float)
    if po.size == 0:
        print("No peak data to plot.")
        return

    idx = np.arange(1, len(po) + 1)
    mean_po = np.mean(po)

    plt.figure(figsize=(8, 4))
    plt.plot(idx, po, marker="o", linestyle="-", label="Peak per step")

    plt.axhline(mean_po, color="tab:purple", linestyle="--", label="Mean")

    plt.xlabel("Step index")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()


def plot_peak_angular_velocity_per_step(
    peak_omega,
    title="(D) Peak angular velocity per step",
):
    """Wrapper for peak angular velocity."""
    plot_peak_value_per_step(
        peak_values=peak_omega,
        title=title,
        ylabel="Peak angular velocity",
    )


def plot_peak_accel_per_step(
    peak_accel,
    title="(D) Peak acceleration per step",
):
    """Wrapper for peak acceleration."""
    plot_peak_value_per_step(
        peak_values=peak_accel,
        title=title,
        ylabel="Peak acceleration",
    )


# ============================================================
# 1×3 multi-axis helpers (for gyro or accel)
# ============================================================
def plot_raw_with_steps_grid(
    signals_dict,
    fs,
    step_indices_dict,
    axis_order=("x", "y", "z"),
    kind="gyro",
    suptitle="(A) Raw signals with step markers – 3 axes",
):
    """
    Create a [1 x 3] figure of raw signals with step markers for three axes.

    Parameters
    ----------
    signals_dict : dict
        Mapping axis key -> 1D signal
        e.g. {'x': gx, 'y': gy, 'z': gz} or {'x': ax, 'y': ay, 'z': az}
    fs : float
        Sampling frequency
    step_indices_dict : dict
        Mapping axis key -> step_indices (for that axis)
    axis_order : tuple
        Order of axes to plot ('x','y','z')
    kind : str
        'gyro' or 'accel' (controls y-label)
    suptitle : str
        Overall figure title
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 4))

    ylabel = "Angular velocity" if kind == "gyro" else "Acceleration"

    for i, ax_key in enumerate(axis_order):
        ax_plot = axs[i]
        if ax_key not in signals_dict:
            ax_plot.text(0.5, 0.5, f"No data for {ax_key}", ha="center", va="center")
            ax_plot.axis("off")
            continue

        sig = np.asarray(signals_dict[ax_key], dtype=float)
        n = len(sig)
        t = np.arange(n) / fs
        steps = np.asarray(step_indices_dict.get(ax_key, []), dtype=int)

        ax_plot.plot(t, sig, label=f"{kind} {ax_key}")

        for k, idx in enumerate(steps):
            color = "tab:blue" if (k % 2 == 0) else "tab:orange"
            if 0 <= idx < n:
                ax_plot.axvline(
                    t[idx],
                    color=color,
                    alpha=0.7,
                    linestyle="--",
                    linewidth=1.0,
                )

        ax_plot.set_title(f"{ax_key}-axis")
        ax_plot.set_xlabel("Time (s)")
        if i == 0:
            ax_plot.set_ylabel(ylabel)
        ax_plot.grid(True, alpha=0.3)

    fig.suptitle(suptitle)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])


def plot_peak_value_per_step_grid(
    peak_values_dict,
    axis_order=("x", "y", "z"),
    kind="gyro",
    suptitle="(D) Peak per step – 3 axes",
):
    """
    Create a [1 x 3] figure of peak-per-step plots for three axes.

    Parameters
    ----------
    peak_values_dict : dict
        Mapping axis key -> 1D array of peak values per step.
    axis_order : tuple
        Order of axes to plot
    kind : str
        'gyro' or 'accel' (controls y-label text)
    suptitle : str
        Overall figure title
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 4))

    ylabel = "Peak angular velocity" if kind == "gyro" else "Peak acceleration"

    for i, ax_key in enumerate(axis_order):
        ax_plot = axs[i]
        if ax_key not in peak_values_dict or peak_values_dict[ax_key] is None:
            ax_plot.text(0.5, 0.5, f"No data for {ax_key}", ha="center", va="center")
            ax_plot.axis("off")
            continue

        po = np.asarray(peak_values_dict[ax_key], dtype=float)
        if po.size == 0:
            ax_plot.text(0.5, 0.5, f"No data for {ax_key}", ha="center", va="center")
            ax_plot.axis("off")
            continue

        idx = np.arange(1, len(po) + 1)
        mean_po = np.mean(po)

        ax_plot.plot(idx, po, marker="o", linestyle="-", label=f"{kind} {ax_key}")
        ax_plot.axhline(mean_po, color="tab:purple", linestyle="--", label="Mean")

        ax_plot.set_title(f"{ax_key}-axis")
        ax_plot.set_xlabel("Step index")
        if i == 0:
            ax_plot.set_ylabel(ylabel)
        ax_plot.grid(True, alpha=0.3)

    fig.suptitle(suptitle)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])
