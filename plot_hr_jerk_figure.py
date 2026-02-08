"""
Standalone script to generate HR vs Jerk scatter plot figure [1x3 subplots]
for behavior classification from IMU data.
"""

import re
from datetime import datetime, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# Analysis functions (inline)
# ============================================================

def harmonic_ratio(signal, fs, num_harmonics=10):
    """
    Computes Harmonic Ratio (HR) for a 1D gyro/accel gait signal.
    """
    signal = np.array(signal)
    signal_d = signal - np.mean(signal)

    N = len(signal_d)
    freqs = np.fft.rfftfreq(N, 1/fs)
    fft_vals = np.abs(np.fft.rfft(signal_d))

    # Estimate step frequency using peak in spectrum
    valid = (freqs > 0.5) & (freqs < 3.5)
    if not np.any(valid):
        return np.nan
    dom_idx = np.argmax(fft_vals[valid])
    step_freq = freqs[valid][dom_idx]

    harmonics_even = []
    harmonics_odd = []

    for h in range(1, num_harmonics + 1):
        target_freq = h * step_freq
        idx = np.argmin(np.abs(freqs - target_freq))
        if h % 2 == 0:
            harmonics_even.append(fft_vals[idx])
        else:
            harmonics_odd.append(fft_vals[idx])

    if sum(harmonics_odd) == 0:
        return np.nan

    return sum(harmonics_even) / sum(harmonics_odd)


def jerk_rms(signal, fs):
    """
    Compute angular jerk RMS from a 1D gyro signal.
    """
    sig = np.array(signal, dtype=float)
    if len(sig) < 2:
        return np.nan
    jerk = np.diff(sig) * fs
    return np.sqrt(np.mean(jerk**2))


# ============================================================
# Configuration
# ============================================================

log_file = "data/trial1.txt"
fs = 50  # Sampling frequency (Hz)

# Behavior windows with time ranges
windows = {
    "calm_sleep": {
        "label": "Calm sleep",
        "start": time(21, 58, 39),
        "end":   time(21, 59, 15),
        "timestamps": [],
        "ax": [], "ay": [], "az": [],
        "gx": [], "gy": [], "gz": [],
    },
    "tossing_turning": {
        "label": "Tossing & turning",
        "start": time(21, 59, 20),
        "end":   time(21, 59, 45),
        "timestamps": [],
        "ax": [], "ay": [], "az": [],
        "gx": [], "gy": [], "gz": [],
    },
    "stumbling": {
        "label": "Stumbling",
        "start": time(22, 0, 10),
        "end":   time(22, 0, 40),
        "timestamps": [],
        "ax": [], "ay": [], "az": [],
        "gx": [], "gy": [], "gz": [],
    },
    "pacing": {
        "label": "Pacing",
        "start": time(22, 0, 59),
        "end":   time(22, 1, 20),
        "timestamps": [],
        "ax": [], "ay": [], "az": [],
        "gx": [], "gy": [], "gz": [],
    },
}

# Color map for behaviors
color_map = {
    "Calm sleep": "tab:blue",
    "Tossing & turning": "tab:orange",
    "Stumbling": "tab:red",
    "Pacing": "tab:green",
}

# Axis display names
axis_labels = {
    "gx": "x axis (roll)",
    "gy": "y axis (pitch)",
    "gz": "z axis (yaw)",
}


# ============================================================
# Parse log file
# ============================================================

with open(log_file, "r") as f:
    for line in f:
        m = re.search(r'\[(\d+:\d+:\d+\.\d+)\].*?"([-0-9.,]+)"', line)
        if not m:
            continue

        t_str = m.group(1)
        csv_str = m.group(2)

        t = datetime.strptime(t_str, "%H:%M:%S.%f").time()
        vals = list(map(float, csv_str.split(",")))
        if len(vals) != 6:
            continue

        ax_val, ay_val, az_val, gx_val, gy_val, gz_val = vals

        for w in windows.values():
            if w["start"] <= t <= w["end"]:
                w["timestamps"].append(t)
                w["ax"].append(ax_val)
                w["ay"].append(ay_val)
                w["az"].append(az_val)
                w["gx"].append(gx_val)
                w["gy"].append(gy_val)
                w["gz"].append(gz_val)


# ============================================================
# Generate figure
# ============================================================

axis_list = ["gx", "gy", "gz"]
window_length_s = 5
step_s = 2.5

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

legend_elements = {}
legend_order = ["Calm sleep", "Tossing & turning", "Stumbling", "Pacing"]
legend_handles = [
    Line2D([0], [0], marker='o', color='w', label=lbl,
           markerfacecolor=color_map[lbl], markersize=8,
           markeredgecolor='k', markeredgewidth=0.3)
    for lbl in legend_order
]

for i, axis in enumerate(axis_list):
    ax_plot = axs[i]

    all_hr = []
    all_jerk = []
    all_colors = []

    for key, w in windows.items():
        if not w.get("timestamps"):
            continue
        if axis not in w:
            continue

        sig = np.array(w[axis], dtype=float)
        n = len(sig)
        if n < fs * 2:
            continue

        win_len = int(window_length_s * fs)
        step = int(step_s * fs)
        starts = range(0, n - win_len + 1, step)

        for start in starts:
            seg = sig[start:start + win_len]
            hr = harmonic_ratio(seg, fs)
            jrk = jerk_rms(seg, fs)

            if np.isnan(hr) or np.isnan(jrk):
                continue

            all_hr.append(hr)
            all_jerk.append(jrk)
            lbl = w["label"]
            col = color_map.get(lbl, "black")
            all_colors.append(col)
            legend_elements[lbl] = col
            plt.legend(handles=legend_handles,
    loc='upper right',
    fontsize=10,
    frameon=True,
    borderpad=0.8)

    # Scatter plot
    ax_plot.scatter(
        all_hr, all_jerk,
        c=all_colors,
        s=40,
        alpha=0.7,
        edgecolor="k",
        linewidth=0.3
    )

    ax_plot.set_title(f"HR vs Jerk - {axis_labels.get(axis, axis)}", fontsize=15)
    ax_plot.set_xlabel("Harmonic Ratio (HR)")
    if i == 0:
        ax_plot.set_ylabel("Jerk RMS")
    ax_plot.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 0.88, 1])
plt.show()
