import matplotlib.pyplot as plt

# Original data (unequal lengths)
Height = [50, 55, 170, 175, 160, 165]
weight = [77, 85, 80, 95, 100, 40, 20, 30, 40, 50]

# Trim to the smallest length to avoid ValueError
n = min(len(Height), len(weight))
if len(Height) != len(weight):
	print(f"Warning: trimming data to {n} points (Height: {len(Height)}, weight: {len(weight)})")

x = Height[:n]
y = weight[:n]

plt.scatter(x, y, color='blue', marker='o')
plt.title("Height vs Weight")
plt.xlabel("Height")
plt.ylabel("Weight")

# Save the plot so the result is visible even when a GUI isn't available
out_path = "scatter_output.png"
plt.savefig(out_path)
print(f"Saved plot to {out_path}")
plt.show()

