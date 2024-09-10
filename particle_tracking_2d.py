import csv
import matplotlib.pyplot as plt

# Initialize lists to hold the X, Y, and Z coordinates
x_coords = []
y_coords = []
z_coords = []
joined = []

id = "6832"
# Read the CSV file
csv_file = './material_flow/'+id+'.csv'

with open(csv_file, mode='r') as file:
    csv_reader = csv.DictReader(file)
    for i, row in enumerate(csv_reader):
        if (i + 1) % 2 == 0:  # Skip every 2nd point
            x_coords.append(float(row['avg(X)']) * 1000)
            y_coords.append(float(row['avg(Y)']) * 1000)
            z_coords.append(float(row['avg(Z)']) * 1000)
            joined.append(float(row['avg(Jointed)']))

# Plot the points in 2D space
plt.figure(figsize=(8, 6))
plt.rcParams.update({'font.size': 12})

# Mark the start point with green and end point with red
plt.scatter(x_coords[0], y_coords[0], marker='o', color='green', s=75, label='Start')
plt.scatter(x_coords[-1], y_coords[-1], marker='o', color='red', s=75, label='Stop')


# Plot the path with different colors based on the 'joined' array
flage =1
flage_2 =1
for i in range(len(x_coords) - 1):
    if joined[i] == 2:
        plt.plot(x_coords[i:i + 2], y_coords[i:i + 2], linestyle='-', color='orange', label='Tracked particle (Deposition)' if flage == 1 else "")
        flage=0
        if i != len(x_coords) - 2:
            plt.scatter(x_coords[i:i + 2], y_coords[i:i + 2], color='orange')
    else:
        plt.plot(x_coords[i:i + 2], y_coords[i:i + 2], linestyle='-', color='blue', markersize=5, label='Tracked particle (Rod)' if flage_2 == 1 else "")
        flage_2=0
        if i != 0: 
        #if i > 0 and i < len(x_coords) - 2:
            plt.scatter(x_coords[i:i + 2], y_coords[i:i + 2], color ="b")





# Add labels and title
plt.xlabel('x [mm]')
plt.ylabel('y [mm]')
plt.ylim((-15,15))
plt.legend()

# Show the plot
plt.grid(True)
plt.savefig("./material_flow/"+id+".jpg", dpi=3000)
plt.savefig("./material_flow/"+id+".pdf")
plt.show()
