import re
from datetime import datetime, time
import matplotlib.pyplot as plt
from analysis.harmonicRatio import harmonic_ratio
from analysis.jerkIndex import jerk_rms
from plot.scatter_hr_jerk import plot_hr_vs_jerk
import numpy as np
from analysis.basicStepAnalysis import (
    detect_steps_from_gyro,
    detect_steps_from_accel,
    compute_step_metrics,
    plot_raw_gyro_with_steps,
    plot_raw_accel_with_steps,
    plot_step_time_series,
    plot_stride_time_series,
    plot_peak_angular_velocity_per_step,
    plot_peak_accel_per_step,
    plot_raw_with_steps_grid,
    plot_peak_value_per_step_grid,
)

log_file = "/Users/ailee/OneDrive/Documents/Github/SeniorDesign/data/trial1.txt"
SHOW_BASIC_PLOTS = True  # Set False to suppress simple per-window plots that were crashing

# Define your windows
windows = {
    "regular_walking": {
        "label": "Regular Walking",
        "start": time(21, 54, 20),
        "end":   time(21, 54, 40),
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

# Parse the log
with open(log_file, "r") as f:
    for line in f:
        # match: [21:40:35.0250] ... "csv,values,here"
        m = re.search(r'\[(\d+:\d+:\d+\.\d+)\].*?"([-0-9.,]+)"', line)
        if not m:
            continue

        t_str = m.group(1)           # '21:40:35.0250'
        csv_str = m.group(2)         # '-0.440,0.404,-0.235,-142.090,-15.442,1.648'

        # parse time
        t = datetime.strptime(t_str, "%H:%M:%S.%f").time()

        # parse CSV into floats
        vals = list(map(float, csv_str.split(",")))
        if len(vals) != 6:
            continue
        ax, ay, az, gx, gy, gz = vals

        # Slot this sample into any matching window(s)
        for w in windows.values():
            if w["start"] <= t <= w["end"]:
                w["timestamps"].append(t)
                w["ax"].append(ax)
                w["ay"].append(ay)
                w["az"].append(az)
                w["gx"].append(gx)
                w["gy"].append(gy)
                w["gz"].append(gz)

# Helper to convert times to relative seconds
def relative_seconds(t_list):
    t0 = datetime.combine(datetime.today(), t_list[0])
    return [
        (datetime.combine(datetime.today(), t) - t0).total_seconds()
        for t in t_list
    ]

fs = 50  # or whatever your IMU runs at

for key, w in windows.items():
    if not w["timestamps"]:
        continue

    gyro_signal = w["gz"]  # choose axis
    hr = harmonic_ratio(gyro_signal, fs)

    print(f"{w['label']} HR = {hr:.3f}")

for key, w in windows.items():
    if not w["timestamps"]:
        continue

    gyro_signal = w["gz"]  # choose axis (gz is usually good)

    hr = harmonic_ratio(gyro_signal, fs)
    jrk = jerk_rms(gyro_signal, fs)

    print(f"{w['label']}: HR = {hr:.3f}, Jerk RMS = {jrk:.3f}")


# x: forward back
# y: left right
# z: up down

axis_list = ["gx", "gy", "gz"]

# Generate [1,3] plots

axis_list = ["gx", "gy", "gz"]
human_axis_map = {
    "gx": "x axis (roll)",
    "gy": "y axis (pitch)",
    "gz": "z axis (yaw)",
}

# color map for behaviors
color_map = {
    "Calm sleep": "tab:blue",
    "Tossing & turning": "tab:orange",
    "Stumbling": "tab:red",
    "Pacing": "tab:green",
}

if SHOW_BASIC_PLOTS:
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Will store legend entries only once
    legend_elements = {}

    for i, axis in enumerate(axis_list):
        ax_plot = axs[i]

        all_hr = []
        all_jerk = []
        all_colors = []
        all_labels = []

        # build feature arrays for this axis
        for key, w in windows.items():
            if not w.get("timestamps"):
                continue
            if axis not in w:
                continue

            sig = np.array(w[axis], dtype=float)
            n = len(sig)
            if n < fs * 2:
                continue

            win_len = int(5.0 * fs)
            step = int(2.5 * fs)

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
                all_labels.append(lbl)
                col = color_map.get(lbl, "black")
                all_colors.append(col)

                # collect legend entries once
                legend_elements[lbl] = col

        # Convert arrays
        all_hr = np.array(all_hr)
        all_jerk = np.array(all_jerk)

        # --- scatter into subplot ---
        ax_plot.scatter(
            all_hr, all_jerk,
            c=all_colors,
            s=40,
            alpha=0.7,
            edgecolor="k",
            linewidth=0.3
        )

        ax_plot.set_title(f"HR vs Jerk – {human_axis_map.get(axis, axis)}", fontsize=12)
        ax_plot.set_xlabel("Harmonic Ratio (HR)")
        if i == 0:
            ax_plot.set_ylabel("Jerk RMS")

        ax_plot.grid(True, alpha=0.3)

    # --- legend for the whole figure (top right) ---
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker='o', color='k', label=lbl,
               markerfacecolor=col, markersize=8, linestyle='')
        for lbl, col in legend_elements.items()
    ]

    fig.legend(
        handles=legend_handles,
        loc='upper right',
        fontsize=10,
        frameon=True,
        borderpad=0.8
    )

    plt.tight_layout(rect=[0.08, 0.02, 0.88, 0.97])
    plt.show()

fs = 50
w = windows["pacing"]
window_label = w["label"]

# gyro signals
gx, gy, gz = w["gx"], w["gy"], w["gz"]
# accel signals
ax, ay, az = w["ax"], w["ay"], w["az"]

# detect steps (e.g., from gz and az)
steps_gz = detect_steps_from_gyro(gz, fs)
steps_az = detect_steps_from_accel(az, fs)

# metrics
metrics_gz = compute_step_metrics(steps_gz, fs, signal=gz)
metrics_az = compute_step_metrics(steps_az, fs, signal=az)

# (A) single-axis
plot_raw_gyro_with_steps(gz, fs, steps_gz, title=f"(A) Gyro gz – {window_label}")
plot_raw_accel_with_steps(az, fs, steps_az, title=f"(A) Accel az – {window_label}")

# (B), (C)
plot_step_time_series(metrics_gz["step_times"], title=f"(B) Step time series – {window_label}")
plot_stride_time_series(metrics_gz["stride_times"], title=f"(C) Stride time series – {window_label}")

# (D)
plot_peak_angular_velocity_per_step(metrics_gz["peak_value"], title=f"(D) Peak angular velocity per step – {window_label}")
plot_peak_accel_per_step(metrics_az["peak_value"], title=f"(D) Peak acceleration per step – {window_label}")

# 1×3 grid for raw gyro
signals_gyro = {"x": gx, "y": gy, "z": gz}
step_indices_gyro = {
    "x": detect_steps_from_gyro(gx, fs),
    "y": detect_steps_from_gyro(gy, fs),
    "z": steps_gz,
}
plot_raw_with_steps_grid(signals_gyro, fs, step_indices_gyro, kind="gyro",
                         suptitle=f"(A) Raw angular velocity with steps – 3 gyro axes ({window_label})")

# 1×3 grid for peak per step (gyro)
peak_gyro = {
    "x": compute_step_metrics(step_indices_gyro["x"], fs, signal=gx)["peak_value"],
    "y": compute_step_metrics(step_indices_gyro["y"], fs, signal=gy)["peak_value"],
    "z": metrics_gz["peak_value"],
}
plot_peak_value_per_step_grid(peak_gyro, kind="gyro",
                              suptitle=f"(D) Peak angular velocity per step – 3 gyro axes ({window_label})")

# Show all figures together
plt.show()
