# RUN THIS ONCE TO READ IN LOG DATA AND (OPTIONAL) GET BASIC PRINTOUT OF RAW DATA 

import re
from datetime import datetime, time
import matplotlib.pyplot as plt
import numpy as np
import config
import constants

log_file = constants.LOG_FILE
BASIC_PLOTS = constants.BASIC_PLOTS

#define windows of time as dictionary of dictionaries
#ignore the weird syntax of datetime, just makes it easier for now
#DO NOTE THAT THIS CURRENTLY ASSUMES ALL MEASUREMENTS ARE TAKEN IN THE SAME DAY (IN 1900 OR SMTH) SO IF WE WANT TO TRACK ACROSS MULTIPLE DAYS IT NEEDS TO CHANGE
windows = {
    "all":{
        "label": "all",
        "start": datetime.strptime("21:54:14", "%H:%M:%S"),
        "end":   datetime.strptime("22:01:20", "%H:%M:%S"),
        "timestamps": [],
        "ax": [], "ay": [], "az": [],
        "gx": [], "gy": [], "gz": [],
    },
    "regular_walking": {
        "label": "Regular Walking",
        "start": datetime.strptime("21:54:20", "%H:%M:%S"),
        "end":   datetime.strptime("21:54:40", "%H:%M:%S"),
        "timestamps": [],
        "ax": [], "ay": [], "az": [],
        "gx": [], "gy": [], "gz": [],
    },
    "calm_sleep": {
        "label": "Calm sleep",
        "start": datetime.strptime("21:58:39", "%H:%M:%S"),
        "end":   datetime.strptime("21:59:15", "%H:%M:%S"),
        "timestamps": [],
        "ax": [], "ay": [], "az": [],
        "gx": [], "gy": [], "gz": [],
    },
    "tossing_turning": {
        "label": "Tossing & turning",
        "start": datetime.strptime("21:59:20", "%H:%M:%S"),
        "end":   datetime.strptime("21:59:45", "%H:%M:%S"),
        "timestamps": [],
        "ax": [], "ay": [], "az": [],
        "gx": [], "gy": [], "gz": [],
    },
    "stumbling": {
        "label": "Stumbling",
        "start": datetime.strptime("22:00:10", "%H:%M:%S"),
        "end":   datetime.strptime("22:00:40", "%H:%M:%S"),
        "timestamps": [],
        "ax": [], "ay": [], "az": [],
        "gx": [], "gy": [], "gz": [],
    },
    "pacing": {
        "label": "Pacing",
        "start": datetime.strptime("22:00:59", "%H:%M:%S"),
        "end":   datetime.strptime("22:01:20", "%H:%M:%S"),
        "timestamps": [],
        "ax": [], "ay": [], "az": [],
        "gx": [], "gy": [], "gz": [],
    },
}

#find gaps of whatever size you want to make sure data is coming through cleanly
#ignore the fact i shouldn't really have a function in here but i kinda don't care
#times = list of timestamps (as seconds)
#time_diff = maximum desired difference between timestamps (ie. if time_diff = 1, gaps of > 1 sec will be highlighted)
def find_gaps(times, time_diff):
    gaps = []
    differences = [(times[i] - times[i-1]) for i in range(1, len(times))]
    for i in range(1, len(differences)):
        if differences[i] > time_diff:
            gaps.append((times[i], times[i+1]))
    return gaps

#read in log file and fill in dictionary
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
        for w in windows.values():
            if (w["start"] <= t and t <= w["end"]):
                w["timestamps"].append(t)
                w["ax"].append(ax)
                w["ay"].append(ay)
                w["az"].append(az)
                w["gx"].append(gx)
                w["gy"].append(gy)
                w["gz"].append(gz)

#quick check for blank windows and to make timestamps more readable 
empty_key = []
for key, w in windows.items():
    if not w["timestamps"]:
        print(f"No samples found for window: {w['label']}")
        empty_key.append(key)
        continue

    t_rel = [(t - w["timestamps"][0]).total_seconds() for t in w["timestamps"]]
    w["t_rel"] = t_rel

#delete empty windows
for key in empty_key:
    del windows[key]

config.set("windows", windows)

#if you want to see plots! (i would only run this the first time you want to see the data)
if BASIC_PLOTS == True:
    for key, w in windows.items():
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        fig.suptitle(f"{w['label']}")

        #accel plots
        ax1 = axes[0]
        ax1.plot(w["t_rel"], w["ax"], label="ax")
        ax1.plot(w["t_rel"], w["ay"], label="ay")
        ax1.plot(w["t_rel"], w["az"], label="az")
        ax1.set_ylabel("Acceleration")
        ax1.legend(loc = "upper left")
        ax1.grid(True)

        #gyro plots
        ax2 = axes[1]
        ax2.plot(w["t_rel"], w["gx"], label="gx")
        ax2.plot(w["t_rel"], w["gy"], label="gy")
        ax2.plot(w["t_rel"], w["gz"], label="gz")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Gyro")
        ax2.legend(loc = "upper left")
        ax2.grid(True)

        #find missing data + highlight in red
        gaps = find_gaps(w["t_rel"], .5)
        if gaps:
            for (start, end) in gaps:
                for ax in axes:  # shade both accel + gyro subplots
                        ax.axvspan(start, end, color="red", alpha=0.15, label="Missing data")
        plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()