import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

df = pd.read_csv("/home/advaitgore/PycharmProjects/torchdisorder/data-release/xrd_measurements/Si/annealed-S-of-Q.csv")   # Your data file
F = df["F"].values

# Smooth the data using a Savitzky-Golay filter
F_smooth = savgol_filter(F, window_length=11, polyorder=2)

# Uncertainty is estimated as the absolute local residual between data and the smooth fit
dF = np.abs(F - F_smooth)

# Set a minimum noise floor to avoid zeros (e.g., 0.01 * mean F)
noise_floor = 0.01 * np.mean(np.abs(F))
dF = np.clip(dF, noise_floor, None)

df["dF"] = dF
df.to_csv("/home/advaitgore/PycharmProjects/torchdisorder/data-release/xrd_measurements/Si/annealed-S-of-Q.csv", index=False)

if main