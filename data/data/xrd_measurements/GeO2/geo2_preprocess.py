import pandas as pd

# Read current file
df = pd.read_csv("/home/advaitgore/PycharmProjects/torchdisorder/data-release/xrd_measurements/GeO2/T_of_r.csv")

# Save WITH index column (pandas default)
df.to_csv("T_of_r.csv")  # Don't use index=False!

# Verify
print(pd.read_csv("T_of_r.csv").head())
