import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
file_path = '/home/advaitgore/PycharmProjects/torchdisorder/data-release/xrd_measurements/NaTaCl6/F_of_Q.csv'
data = pd.read_csv(file_path)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(data['Q'], data['F'], linewidth=1.5, color='blue')
plt.xlabel('Q', fontsize=12)
plt.ylabel('F', fontsize=12)
plt.title('Q vs F Plot', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
plt.savefig('Q_vs_F_plot.png', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()

print("Plot saved as 'Q_vs_F_plot.png'")
