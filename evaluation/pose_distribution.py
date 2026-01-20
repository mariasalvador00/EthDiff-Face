import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("ethdiff_face_landmarks.csv")

# --- Pose score ---
df["eye_mid_x"] = (df["le_x"] + df["re_x"]) / 2.0
df["eye_dist"] = np.abs(df["re_x"] - df["le_x"])
df["pose_score"] = (df["nose_x"] - df["eye_mid_x"]) / df["eye_dist"]

# Clip extreme outliers for readability
df = df[df["pose_score"].between(-0.6, 0.6)]

# --- Plot ---
fig, ax = plt.subplots(figsize=(8.5, 5))

bins = np.linspace(-0.6, 0.6, 50)

colors = {
    "african": "#164AF4",
    "asian": "#DCF30E",
    "caucasian": "#04BA74",
    "indian": "#FD0303",
}

for eth in sorted(df["ethnicity"].unique()):
    ax.hist(
        df[df["ethnicity"] == eth]["pose_score"],
        bins=bins,
        histtype="step",
        linewidth=2.0,
        label=eth.capitalize(),
        color=colors.get(eth)
    )

# Frontal pose threshold
POSE_THRESHOLD = 0.15
ax.axvline(POSE_THRESHOLD, color="gray", linestyle="--", linewidth=1)
ax.axvline(-POSE_THRESHOLD, color="gray", linestyle="--", linewidth=1)

ax.set_xlabel("yaw")
ax.set_ylabel("number of imgs")
ax.set_title("yaw")

ax.legend(
    title="Ethnicity",
    frameon=False,
    fontsize=9,
    title_fontsize=9,
    loc="upper right",
    bbox_to_anchor=(0.98, 0.98),
    bbox_transform=ax.transAxes
)




plt.subplots_adjust(top=0.85, bottom=0.22)

plt.savefig(
    "plots/yaw_histogram_per_ethnicity.png",
    dpi=300
)


plt.close(fig)

pose_stats = (
    df.groupby("ethnicity")["pose_score"]
      .agg(["mean", "std", "count"])
      .reset_index()
)

print(pose_stats)
