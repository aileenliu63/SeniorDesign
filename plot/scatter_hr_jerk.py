import numpy as np
import matplotlib.pyplot as plt

def _segment_indices(n_samples, fs, window_length_s, step_s):
    """
    Helper: yields (start, end) sample indices for sliding windows.
    """
    win_len = int(window_length_s * fs)
    step = int(step_s * fs)
    if win_len <= 0 or step <= 0:
        return

    start = 0
    while start + win_len <= n_samples:
        yield start, start + win_len
        start += step


def plot_hr_vs_jerk(
    windows,
    fs,
    harmonic_ratio_fn,
    jerk_fn,
    axis_name="gz",
    window_length_s=4.0,
    step_s=2.5,
    min_window_length_s=2.0,   # <- NEW: minimum length we’ll accept
    color_map=None,
    annotate_centroids=True,
):
    """
    Generate a scatter plot of Harmonic Ratio (HR) vs Jerk RMS
    using multiple sliding subwindows within each labeled behavior window.
    More forgiving: if a window is too short for window_length_s,
    but at least min_window_length_s long, we just use the whole thing.
    """

    # Default colors if none provided
    default_colors = {
        "Calm sleep": "tab:blue",
        "Tossing & turning": "tab:orange",
        "Stumbling": "tab:red",
        "Pacing": "tab:green",
    }
    if color_map is None:
        color_map = default_colors

    all_hr = []
    all_jerk = []
    all_labels = []
    all_colors = []

    # For centroid computation
    label_hr = {}
    label_jerk = {}

    min_len_samples = int(min_window_length_s * fs)

    # --- Loop over top-level behavior windows ---
    for key, w in windows.items():
        if not w.get("timestamps"):
            continue

        if axis_name not in w:
            print(f"Warning: axis '{axis_name}' not in window '{key}'. Skipping.")
            continue

        signal = np.array(w[axis_name], dtype=float)
        n = len(signal)
        label = w.get("label", key)

        if n < min_len_samples:
            print(f"{label}: only {n/fs:.2f}s of data on {axis_name}, skipping (too short).")
            continue

        # Try normal sliding segmentation
        indices = list(_segment_indices(n, fs, window_length_s, step_s))

        # If nothing fits, fall back to “use the whole segment”
        if not indices:
            print(f"{label}: segment too short for {window_length_s}s, "
                  f"using entire {n/fs:.2f}s on {axis_name}.")
            indices = [(0, n)]

        seg_count = 0
        for start, end in indices:
            seg = signal[start:end]

            # Compute features for this subwindow
            hr = harmonic_ratio_fn(seg, fs)
            jrk = jerk_fn(seg, fs)

            if np.isnan(hr) or np.isnan(jrk):
                continue

            all_hr.append(hr)
            all_jerk.append(jrk)
            all_labels.append(label)
            all_colors.append(color_map.get(label, "black"))

            label_hr.setdefault(label, []).append(hr)
            label_jerk.setdefault(label, []).append(jrk)

            seg_count += 1

        print(f"{label}: generated {seg_count} subwindows for HR/Jerk on {axis_name}.")

    if not all_hr:
        print("No subwindows generated; check window_length_s, step_s, and data length.")
        return

    all_hr = np.array(all_hr)
    all_jerk = np.array(all_jerk)

    # --- Scatter plot ---
    plt.figure(figsize=(7, 6))
    plt.scatter(all_hr, all_jerk, c=all_colors, s=40, alpha=0.7,
                edgecolor="k", linewidth=0.3)

    # Optionally annotate centroids for each label
    if annotate_centroids:
        for label in label_hr.keys():
            hr_mean = np.mean(label_hr[label])
            jrk_mean = np.mean(label_jerk[label])
            plt.scatter(hr_mean, jrk_mean, marker="X", s=120,
                        c=color_map.get(label, "black"), edgecolor="k", linewidth=1.0)
            plt.annotate(
                label,
                (hr_mean, jrk_mean),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=18,
                fontweight="bold",
            )
    # Axis label mapping
    axis_label_map = {
        "gx": "x axis",
        "gy": "y axis",
        "gz": "z axis",
    }

    axis_label = axis_label_map.get(axis_name, axis_name)

    plt.xlabel("Harmonic Ratio (HR)")
    plt.ylabel("Jerk RMS")
    plt.title(f"Stability Feature Space – {axis_label}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
