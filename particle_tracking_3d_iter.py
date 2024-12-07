import csv
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


# Initialize lists to hold the X, Y, and Z coordinates
x_coords = []
y_coords = []
z_coords = []
joined = []
pl = []

id = "6832_new"
#id = "6638"
# Read the CSV file
csv_file = './material_flow/'+id+'.csv'

with open(csv_file, mode='r') as file:
    csv_reader = csv.DictReader(file)
    for i, row in enumerate(csv_reader):
        if (i + 1) % 1 == 0:  # Skip every 2nd point
            x_coords.append(float(row['avg(X)']) * -1000)
            y_coords.append(float(row['avg(Y)']) * 1000)
            z_coords.append(float(row['avg(Z)']) * -1000)
            #pl.append(float(row['avg(Plastic_Strain)']))
            joined.append(float(row['avg(Jointed)']))

# Plot the points in 3D space
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
plt.rcParams.update({'font.size': 20})

# Make the borders of the 3D box bold
ax.w_xaxis.line.set_linewidth(2)
ax.w_yaxis.line.set_linewidth(2)
ax.w_zaxis.line.set_linewidth(2)

# Create a parameter t for interpolation
t = np.linspace(0, 1, len(x_coords))

# Create a spline of the data
spl_x = make_interp_spline(t, x_coords, k=2)
spl_y = make_interp_spline(t, y_coords, k=2)
spl_z = make_interp_spline(t, z_coords, k=2)

# Create a spline for the joined data
spl_joined = make_interp_spline(t, joined, k=0)

# Generate new points for a smoother curve
t_new = np.linspace(0, 1, 1000)
x_smooth = spl_x(t_new)
y_smooth = spl_y(t_new)
z_smooth = spl_z(t_new)

# Generate new points for the joined data
joined_smooth = spl_joined(t_new)

# Mark the start point with green and end point with red
ax.scatter(x_coords[0], y_coords[0], z_coords[0], marker='o', color='green')

first_rod = True
first_deposition = True
for i in range(len(joined_smooth) - 1):
    if joined_smooth[i] == 0:
        if first_rod:
            ax.plot(x_smooth[i:i+2], y_smooth[i:i+2], z_smooth[i:i+2], color='black', alpha=1)
            first_rod = False
        else:
            ax.plot(x_smooth[i:i+2], y_smooth[i:i+2], z_smooth[i:i+2], color='black', alpha=1)
    else:
        if first_deposition:
            ax.plot(x_smooth[i:i+2], y_smooth[i:i+2], z_smooth[i:i+2], color='black', alpha=1)
            first_deposition = False
        else:
            ax.plot(x_smooth[i:i+2], y_smooth[i:i+2], z_smooth[i:i+2], color='black', alpha=1)

for i in range(len(joined_smooth) - 1):
    if joined_smooth[i] == 2:
        if i % 20 == 0:
            ax.scatter(x_smooth[i], y_smooth[i], z_smooth[i], color="r", alpha=1)
    if joined_smooth[i] == 0:
        if i % 10 == 0:
            ax.scatter(x_smooth[i], y_smooth[i], z_smooth[i], color='blue', alpha=1)

            

ax.scatter(x_coords[10], y_coords[10], z_coords[10], marker='o', color='blue')
ax.scatter(x_coords[-1], y_coords[-1], z_coords[-1], marker='o', color='red')



# Add labels and title
""" ax.set_xlabel('x [mm]')
ax.set_ylabel('y [mm]')
ax.set_zlabel('z [mm]') """
#plt.ylim((-5,5))

#plt.legend()

# Show the plot
plt.grid(True)
plt.savefig("./material_flow/"+id+".jpg", dpi=2000, bbox_inches='tight')
plt.savefig("./material_flow/"+id+".pdf")
plt.show()
