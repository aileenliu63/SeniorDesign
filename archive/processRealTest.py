import argparse
import re
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from analysis.harmonicRatio import harmonic_ratio
from analysis.jerkIndex import jerk_rms
from plot.scatter_hr_jerk import plot_hr_vs_jerk


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse BLE log and plot acceleration axes (relative time)."
    )
    parser.add_argument(
        "--file",
        default="data/realTest3.txt",
        help="Path to log file (default: data/realTest2.txt)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.file)
    dataset_label = data_path.stem  # e.g., testData or realTest

    # Parse the log and collect samples
    t_list = []
    ax, ay, az = [], [], []
    gx, gy, gz = [], [], []

    with open(data_path, "r") as f:
        for line in f:
            # Example: [21:40:11.8660] Application: "0.004,-0.036,1.026,-0.244,-0.610,-0.488" value received.
            m = re.search(r'\[(\d+:\d+:\d+\.\d+)\].*?"([-0-9.,]+)"', line)
            if not m:
                continue

            t_str = m.group(1)
            csv_str = m.group(2)

            # Parse timestamp and CSV
            try:
                t = datetime.strptime(t_str, "%H:%M:%S.%f")
                vals = list(map(float, csv_str.split(",")))
                if len(vals) != 6:
                    continue
            except ValueError:
                continue

            ax_val, ay_val, az_val, gx_val, gy_val, gz_val = vals
            t_list.append(t)
            ax.append(ax_val)
            ay.append(ay_val)
            az.append(az_val)
            gx.append(gx_val)
            gy.append(gy_val)
            gz.append(gz_val)

    if not t_list:
        print(f"No samples found in {data_path}")
        raise SystemExit

    # Convert timestamps to relative seconds from first sample
    t0 = t_list[0]
    t_rel = [(ts - t0).total_seconds() for ts in t_list]

    # Restrict to [33, 44] seconds relative
    mask = [33.0 <= t <= 44.0 for t in t_rel]
    if not any(mask):
        print(f"No samples in 33–44s window for {data_path}")
        raise SystemExit

    t_rel_trim = [t for t, keep in zip(t_rel, mask) if keep]
    ax_trim = [v for v, keep in zip(ax, mask) if keep]
    ay_trim = [v for v, keep in zip(ay, mask) if keep]
    az_trim = [v for v, keep in zip(az, mask) if keep]
    gx_trim = [v for v, keep in zip(gx, mask) if keep]
    gy_trim = [v for v, keep in zip(gy, mask) if keep]
    gz_trim = [v for v, keep in zip(gz, mask) if keep]

    # Estimate sampling frequency from trimmed timestamps (median dt)
    fs = 50  # fallback default
    if len(t_rel_trim) >= 2:
        dt = np.median(np.diff(t_rel_trim))
        if dt > 0:
            fs = 1.0 / dt
    print(f"Estimated fs ≈ {fs:.2f} Hz for {dataset_label}")

    # Plot acceleration [1 x 3]
    fig, axs = plt.subplots(1, 3, figsize=(18, 4), sharex=True)
    axes_data = [("ax", ax_trim), ("ay", ay_trim), ("az", az_trim)]

    for i, (label, data) in enumerate(axes_data):
        axs[i].plot(t_rel_trim, data, label=label)
        axs[i].set_title(f"{label} vs time ({dataset_label})")
        axs[i].set_xlabel("Time (s)")
        if i == 0:
            axs[i].set_ylabel("Acceleration")
        axs[i].grid(True, alpha=0.3)

    fig.suptitle(f"Acceleration – {dataset_label} (33–44 s, relative time)")
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])
    plt.show()

    # Plot gyro [1 x 3]
    fig2, axs2 = plt.subplots(1, 3, figsize=(18, 4), sharex=True)
    gyro_data = [("gx", gx_trim), ("gy", gy_trim), ("gz", gz_trim)]

    for i, (label, data) in enumerate(gyro_data):
        axs2[i].plot(t_rel_trim, data, label=label, color="tab:red")
        axs2[i].set_title(f"{label} vs time ({dataset_label})")
        axs2[i].set_xlabel("Time (s)")
        if i == 0:
            axs2[i].set_ylabel("Gyro")
        axs2[i].grid(True, alpha=0.3)

    fig2.suptitle(f"Gyro – {dataset_label} (33–44 s, relative time)")
    fig2.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])
    plt.show()

    # HR vs Jerk scatter side-by-side for gx, gy, gz
    axis_map = {"gx": gx_trim, "gy": gy_trim, "gz": gz_trim}
    win_len_s = 4.0
    step_s = 2.0
    min_len_s = 2.0
    min_len_samples = int(min_len_s * fs)

    fig3, axs3 = plt.subplots(1, 3, figsize=(18, 5))
    for i, (axis_name, sig) in enumerate(axis_map.items()):
        ax_plot = axs3[i]
        sig_arr = np.array(sig, dtype=float)
        n = len(sig_arr)
        if n < min_len_samples:
            ax_plot.text(0.5, 0.5, f"Too short for {axis_name}", ha="center", va="center")
            ax_plot.axis("off")
            continue

        # Build subwindows
        win_len = int(win_len_s * fs)
        step = int(step_s * fs)
        starts = range(0, max(n - win_len + 1, 1), step)
        if n < win_len:
            starts = [0]  # fall back to full segment
            win_len = n

        hr_vals, jerk_vals = [], []
        for start in starts:
            end = min(start + win_len, n)
            seg = sig_arr[start:end]
            hr_vals.append(harmonic_ratio(seg, fs))
            jerk_vals.append(jerk_rms(seg, fs))

        hr_vals = np.array(hr_vals)
        jerk_vals = np.array(jerk_vals)
        mask_finite = np.isfinite(hr_vals) & np.isfinite(jerk_vals)
        hr_vals = hr_vals[mask_finite]
        jerk_vals = jerk_vals[mask_finite]

        if hr_vals.size == 0:
            ax_plot.text(0.5, 0.5, f"No HR/Jerk for {axis_name}", ha="center", va="center")
            ax_plot.axis("off")
            continue

        ax_plot.scatter(hr_vals, jerk_vals, c="tab:blue", s=40, alpha=0.7,
                        edgecolor="k", linewidth=0.3)
        ax_plot.set_title(f"HR vs Jerk – {axis_name} ({dataset_label})")
        ax_plot.set_xlabel("Harmonic Ratio (HR)")
        if i == 0:
            ax_plot.set_ylabel("Jerk RMS")
        ax_plot.grid(True, alpha=0.3)

    fig3.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
