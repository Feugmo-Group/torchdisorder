import pandas as pd
import matplotlib.pyplot as plt


def plot_csv_data(csv_file_path, x_label, y_label, title):
    # Load CSV into DataFrame
    data = pd.read_csv(csv_file_path, header=None)

    # Assuming two columns (x and y)
    x = data.iloc[:, 0]
    y = data.iloc[:, 1]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker='o', linestyle='-', color='blue')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.show()


# Example usage:
plot_csv_data(
    "/home/advaitgore/PycharmProjects/torchdisorder/data-release/xrd_measurements/Si/annealed-S-of-Q.csv",
    x_label="Q (1/Angstrom)",
    y_label="S(Q)",
    title="Structure Factor S(Q) for Si"
)

plot_csv_data(
    "/home/advaitgore/PycharmProjects/torchdisorder/data-release/xrd_measurements/Si/annealed-J-of-R.csv",
    x_label="r (Angstrom)",
    y_label="J(r)",
    title="J(r) for Si"
)
