import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d


path = './1200-6/1200-6_0.csv'  #set A
df = pd.read_csv(path)
df["Points:0"] *= 1000
df["Points:1"] *= 1000
df["Points:2"] *= 1000
df["Temperature"] -= 273
#df = df.drop(df[df['Points:2'] < 0].index)

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the points
#ax.scatter(df["Points:0"], df["Points:1"], df["Points:2"], c='b', marker='.', alpha=0.1)

# Parameters for cells
cell_width = 8
cell_length = df["Points:0"].max() - df["Points:0"].min()  # Entire length in x direction
cell_height = df["Points:2"].max() - df["Points:2"].min()  # Entire thickness in z direction

# Create cells
y_min = df["Points:1"].min()
y_max = df["Points:1"].max()

for y in range(int(y_min), int(y_max), cell_width):
    ax.bar3d(
        df["Points:0"].min(), y, df["Points:2"].min(),  # bottom corner of the cell
        cell_length, cell_width, cell_height,  # dimensions of the cell
        color='b', edgecolor='black'  # appearance of the cell
    )

# Set equal scaling by adjusting the limits
max_range = max(df.max() - df.min()) / 2.0

mid_x = (df["Points:0"].max() + df["Points:0"].min()) * 0.5
mid_y = (df["Points:1"].max() + df["Points:1"].min()) * 0.5
mid_z = (df["Points:2"].max() + df["Points:2"].min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# Add labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label') 
""" ax.grid(False)
ax.set_axis_off() """
# Show the plot
#plt.savefig("ps_temp.jpg", dpi=3000)
plt.show()
