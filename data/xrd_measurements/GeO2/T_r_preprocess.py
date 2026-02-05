import numpy as np
import pandas as pd

# Read GeO2 T(r) data
input_file = "/home/advaitgore/PycharmProjects/torchdisorder/data-release/xrd_measurements/GeO2/T_r"
data = np.loadtxt(input_file, skiprows=3)

r = data[:, 0]
T_r = data[:, 1]

# Set cutoff before first Ge-O peak (~1.7-1.8 Å)
r_cutoff = 1.5  # Conservative cutoff
num_zeroed = np.sum(r < r_cutoff)

print(f"Zeroing {num_zeroed} points with r < {r_cutoff} Å")

# Zero T(r) values before first large spike
T_r[r < r_cutoff] = 0.0

# Save to CSV
output = pd.DataFrame({'r': r, 'T': T_r})
output.to_csv("T_of_r.csv", index=False)

print("Preprocessing complete!")
