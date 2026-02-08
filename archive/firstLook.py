import re
from datetime import datetime, time
import matplotlib.pyplot as plt
import numpy as np

def find_gaps(times, time_diff):
    gaps = []
    differences = [(times[i] - times[i-1]) for i in range(1, len(times))]
    for i in range(1, len(differences)):
        if differences[i] > time_diff:
            gaps.append((times[i], times[i+1]))
    return gaps

log_file = "/Users/ailee/OneDrive/Documents/Github/SeniorDesign/data/trial1.txt"

timestamps = []
axs = []
ays = []
azs = []
gxs = []
gys = []
gzs = []

with open(log_file, "r") as f:
    for line in f:
        # match: [21:40:35.0250] ... "csv,values,here"
        m = re.search(r'\[(\d+:\d+:\d+\.\d+)\].*?"([-0-9.,]+)"', line)
        if not m:
            continue

        t_str = m.group(1)           # '21:40:35.0250'
        csv_str = m.group(2)         # '-0.440,0.404,-0.235,-142.090,-15.442,1.648'

        # parse time
        t = datetime.strptime(t_str, "%H:%M:%S.%f")

        # parse CSV into floats
        vals = list(map(float, csv_str.split(",")))
        if len(vals) != 6:
            continue
        ax, ay, az, gx, gy, gz = vals

        # Slot this sample into any matching window(s)
        timestamps.append(t)
        axs.append(ax)
        ays.append(ay)
        azs.append(az)
        gxs.append(gx)
        gys.append(gy)
        gzs.append(gz)
print(t)

rel_timestamps = [(t - timestamps[0]).total_seconds() for t in timestamps]

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
fig.suptitle("All Data with Activity Highlights")

ax1, ax2 = axes

#accel plots
ax1.plot(rel_timestamps, axs, label="ax")
ax1.plot(rel_timestamps, ays, label="ay")
ax1.plot(rel_timestamps, azs, label="az")
ax1.set_ylabel("Acceleration")
ax1.grid(True)
ax1.legend(loc="upper left")

#gyro plots
ax2.plot(rel_timestamps, gxs, label="gx")
ax2.plot(rel_timestamps, gys, label="gy")
ax2.plot(rel_timestamps, gzs, label="gz")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Gyro")
ax2.grid(True)
ax2.legend(loc="upper left")

#find missing data
gaps = find_gaps(rel_timestamps, .5)
print(gaps)

if gaps:
    for (start, end) in gaps:
        for ax in axes:  # shade both accel + gyro subplots
                ax.axvspan(start, end, color="red", alpha=0.15, label="Missing data")

plt.show()