import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read the CSV file
file_path = './material_flow/6637.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

# Extract X, Y, Z coordinates
X = df['avg(X)']
Y = df['avg(Y)']
Z = df['avg(Z)']

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the points and connect them with lines
ax.plot(X, Y, Z, marker='o', linestyle='-', color='b')

# Set labels
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

# Show the plot
plt.show()
