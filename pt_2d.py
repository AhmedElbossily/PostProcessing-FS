import csv
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np


# Initialize lists to hold the X, Y, and Z coordinates
x_coords = []
y_coords = []
z_coords = []
joined = []
pl = []

id = "6832_new_new_new"
id = "6973"
#id = "6832_new"
#id = "6713"
#id = "6638"
plt.rcParams.update({'font.size': 20})
plt.figure(figsize=(10, 10))
plt.ylim((-7,7))
plt.xlim((-7,13))

# Read the CSV file
csv_file = './material_flow/'+id+'.csv'

with open(csv_file, mode='r') as file:
    csv_reader = csv.DictReader(file)
    for i, row in enumerate(csv_reader):
        if (i + 1) % 1 == 0:  # Skip every 2nd point
            x_coords.append(float(row['avg(X)']) * -1000)
            y_coords.append(float(row['avg(Y)']) * 1000)
            z_coords.append(float(row['avg(Z)']) * 1000)
            #pl.append(float(row['avg(Plastic_Strain)']))
            joined.append(float(row['avg(Jointed)']))

# Plot the points in 2D space
#plt.figure(figsize=(10, 4.4))



# Create a parameter t for interpolation
t = np.linspace(0, 1, len(x_coords))

# Create a spline of the data
spl_x = make_interp_spline(t, x_coords, k=2)
spl_y = make_interp_spline(t, y_coords, k=2)
spl_z = make_interp_spline(t, z_coords, k=2)
spl_joined = make_interp_spline(t, joined, k=0)

# Generate new points for a smoother curve
t_new = np.linspace(0, 1, 10000)
x_smooth = spl_x(t_new)
y_smooth = spl_y(t_new)
z_smooth = spl_z(t_new)

# Generate new points for the joined data
joined_smooth = spl_joined(t_new)

# Mark the start point with green and end point with red
plt.scatter(x_coords[0], y_coords[0], marker='o', color='green')  # Removed label
plt.scatter(x_coords[-1], y_coords[-1], marker='o', color='r')

first_rod = True
first_deposition = True
for i in range(len(joined_smooth) - 1):
    if joined_smooth[i] == 0:
        if first_rod:
            plt.plot(x_smooth[i:i+2], y_smooth[i:i+2], color='black', alpha=1)  # Removed label
            first_rod = False
        else:
            plt.plot(x_smooth[i:i+2], y_smooth[i:i+2], color='black', alpha=1)
    else:
        if first_deposition:
            plt.plot(x_smooth[i:i+2], y_smooth[i:i+2], color='black', alpha=1)
            first_deposition = False
        else:
            plt.plot(x_smooth[i:i+2], y_smooth[i:i+2], color='black', alpha=1)

""" for i in range(len(joined_smooth) - 1):
    if joined_smooth[i] == 2:
        if i % 20 == 0:
            plt.scatter(x_smooth[i], y_smooth[i], color="r", alpha=1)
    if joined_smooth[i] == 0:
        if i % 5 == 0:
            plt.scatter(x_smooth[i], y_smooth[i], color='blue', alpha=1) """

for i in range(len(joined) - 1):
    if joined[i] == 2:
        if i % 1 == 0:
            plt.scatter(x_coords[i], y_coords[i], color="r", alpha=1)
    if joined[i] == 0:
        if i % 2 == 0:
            plt.scatter(x_coords[i], y_coords[i], color='blue', alpha=1)

plt.scatter(x_smooth[10], y_smooth[10], color='blue', alpha=1)  # Removed label
plt.scatter(x_smooth[-1], y_smooth[-1], color='r', alpha=1)  # Removed label

# Add labels and title
#plt.xlabel('x [mm]')
#plt.ylabel('y [mm]')

#plt.legend()  # Commented out the legend

# Show the plot
plt.grid(True)
plt.savefig("./material_flow/"+id+".jpg", dpi=300)
plt.savefig("./material_flow/"+id+".pdf")
plt.show()
