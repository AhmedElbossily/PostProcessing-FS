import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as st
from scipy.interpolate import interp1d

# Define the file paths
""" paths = [
    "./deposition/1200-4_0.75.csv",
    "./deposition/1200-6_0.75.csv",
    "./deposition/1200-8_0.75.csv",
    "./deposition/1500-6_0.75.csv",
    "./deposition/0900-6_0.75.csv",
    #"./deposition/900-6_0.65.csv"
] """

paths = [
    "./deposition/1200-4_0.75.csv",
    "./deposition/1200-6_0.75.csv",
    "./deposition/1200-8_0.75.csv",
    "./deposition/1500-6_0.75.csv",
    "./deposition/0900-6_0.75.csv"
]

t_1200_4=[]
t_1200_6=[]
t_1200_8=[]
t_1500_6=[]
t_0900_6=[]
t_0600_6=[]
# Generate the X and Y grid
min_x, max_x = 20, 55
min_y, max_y = -4, 8


""" min_x, max_x = 36, 38
min_y, max_y = -5, 9 """

def calculate_thickness_in_cells(x_coords, y_coords, z_coords, cell_size=(2, 2, 3)):
    # Ensure that x_coords, y_coords, and z_coords have the same length
    if not (len(x_coords) == len(y_coords) == len(z_coords)):
        raise ValueError("x_coords, y_coords, and z_coords must have the same length")
    
    # Convert coordinates to numpy arrays
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    z_coords = np.array(z_coords)
    
    # Find the minimum and maximum coordinates
    min_z, max_z = 0, 3
    
    # Calculate the number of cells in each dimension
    num_cells_x = int(np.ceil((max_x - min_x) / cell_size[0]))
    num_cells_y = int(np.ceil((max_y - min_y) / cell_size[1]))
    num_cells_z = int(np.ceil((max_z - min_z) / cell_size[2]))
    
    # Initialize a list to store thicknesses of all cells
    cell_thicknesses = []
    # Initialize a 2D array to store the point count in each cell
    cell_counts = np.zeros((num_cells_x, num_cells_y))
    
    # Loop through each cell
    for i in range(num_cells_x):
        for j in range(num_cells_y):
            for k in range(num_cells_z):
                # Determine the boundaries of the current cell
                x_start = min_x + i * cell_size[0]
                x_end = x_start + cell_size[0]
                y_start = min_y + j * cell_size[1]
                y_end = y_start + cell_size[1]
                z_start = min_z + k * cell_size[2]
                z_end = z_start + cell_size[2]
                
                # Find points within the current cell
                in_cell = (
                    (x_coords >= x_start) & (x_coords < x_end) &
                    (y_coords >= y_start) & (y_coords < y_end) &
                    (z_coords >= z_start) & (z_coords < z_end)
                )
                
                # Extract Z coordinates of points within the current cell
                z_coords_in_cell = z_coords[in_cell]
                
                # Calculate the thickness in the Z direction for the current cell
                if len(z_coords_in_cell) > 0:
                    min_z_in_cell = 0
                    max_z_in_cell = np.max(z_coords_in_cell)
                    thickness = max_z_in_cell - min_z_in_cell
                else:
                    thickness = 0  # If no points in the cell, thickness is 0
                
                # Store the thickness
                cell_thicknesses.append(thickness)
                cell_counts[i, j] = thickness
    
    return cell_counts, cell_thicknesses

def save_to_array(arry, min, avr,max):
    arry.append(min)
    arry.append(avr)
    arry.append(max)
    return arry

# Iterate over each file path
for path in paths:
    print("************************")
    print(f"Processing {path[-15:]}")
    df = pd.read_csv(path)
    df["Points:0"] *= 1000
    df["Points:0"] += 60
    df["Points:1"] *= -1000
    df["Points:2"] *= -1000


    # Extract coordinates
    x_coords = df["Points:0"].to_numpy()
    y_coords = df["Points:1"].to_numpy()
    z_coords = df["Points:2"].to_numpy()

    # Calculate thickness in cells
    cell_counts, cell_thicknesses = calculate_thickness_in_cells(x_coords, y_coords, z_coords, cell_size=(2, 2, 3.2))
    
    # Save cell_thicknesses to a CSV file
    csv_file_path = "./deposition/All_thicknesses.csv"
    mode = 'w' if path == paths[0] else 'a'
    header = ['fileName'] + [f'th_{i+1}' for i in range(len(cell_thicknesses))]
    
    with open(csv_file_path, mode) as csv_file:
        if mode == 'w':
            csv_file.write(','.join(header) + '\n')
        csv_file.write(f"{os.path.basename(path).replace('.csv', '')}," + ','.join(map(str, cell_thicknesses)) + '\n')
    
    
    average_thickness = np.average(cell_thicknesses)
    # Check if any element in cell_thicknesses array is less than 0.1 and replace it with average_thickness
    cell_thicknesses = np.array(cell_thicknesses)
    cell_thicknesses = np.where(cell_thicknesses < 0.1, average_thickness, cell_thicknesses)
    median_thickness = np.median(cell_thicknesses)
    # creating a 2-D array using numpy package
    arr = np.array(cell_thicknesses)
    mode_thickness = st.mode(arr)

    min_thickness = np.min(cell_thicknesses)
    max_thickness = np.max(cell_thicknesses)

    print("min_thickness:", min_thickness)
    print("average_thickness:", average_thickness)
    print("median_thickness:", median_thickness)
    print("mode_thickness:", mode_thickness)
    print("max_thickness:", max_thickness)

    if path[-15:-9] == "1200-4":
       t_1200_4= save_to_array(t_1200_4, min_thickness, average_thickness,max_thickness)
    
    if path[-15:-9] == "1200-6":
        t_1200_6= save_to_array(t_1200_6, min_thickness, average_thickness,max_thickness)

    if path[-15:-9] == "1200-8":
        t_1200_8 = save_to_array(t_1200_8, min_thickness, average_thickness,max_thickness)

    if path[-15:-9] == "0900-6":
        t_0900_6 = save_to_array(t_0900_6, min_thickness, average_thickness,max_thickness)

    if path[-15:-9] == "0600-6":
        t_0600_6 = save_to_array(t_0600_6, min_thickness, average_thickness,max_thickness)
            
    if path[-15:-9] == "1500-6":
        t_1500_6 = save_to_array(t_1500_6, min_thickness, average_thickness,max_thickness)
       
    # Save thickness results to a text file
    text_file_path = f"./deposition/thickness/{path[-15:-4]}.txt"
    with open(text_file_path, "w") as text_file:
        text_file.write(f"min_thickness: {min_thickness}\n")
        text_file.write(f"average_thickness: {average_thickness}\n")
        text_file.write(f"max_thickness: {max_thickness}\n")

    # Plotting
    fig, ax = plt.subplots(figsize=(13, 5))
    #plt.rcParams.update({'font.size': 20})
    #plt.rcParams["figure.figsize"] = (13,5)

    #plt.title(f"Deposition Thickness Distribution: {path[-15:-4]}")
    # Scatter plot overlay
    ax.scatter(df["Points:0"].to_numpy(), df["Points:1"].to_numpy(), marker="o", color="royalblue", s=300)

    x_edges = np.arange(min_x, max_x + 2, 2)
    y_edges = np.arange(min_y, max_y + 2, 2)
    X, Y = np.meshgrid(x_edges, y_edges)
    # Plot the grid with thickness using pcolormesh
    cax = ax.pcolormesh(X, Y, cell_counts.T, cmap='viridis', shading='auto')
    # Adding color bar
    #fig.colorbar(cax, ax=ax,fraction=0.046, pad=0).set_label("Cell thickness [mm]")
    fig.colorbar(cax, ax=ax,fraction=0.046, pad=0)
    # Adding labels and title
    #ax.set_xlabel("Deposition length, x direction [mm]")
    #ax.set_ylabel("Deposition width, y direction [mm]")

    # Add arrow and annotations
    ax.arrow(x=7, y=0, dx=55, dy=0, width=.2, color="black")
    #ax.text(20, 7, 'AS')
    #ax.text(20, -7, 'RS')
    #ax.text(15, 1, 'Deposition')
    #ax.text(15, -2, 'Direction')

    plt.tight_layout()
    #plt.savefig(f"./deposition/output_plt/thickness_validation_{path[-15:-4]}.jpg", dpi=300, bbox_inches='tight')
    #plt.show()

marker_list = ["*","s","*"]
plt.figure()
plt.rcParams.update({'font.size': 15})

""" for i in range(0,3):
    plt.plot(4,t_1200_4[i],marker=marker_list[i], c="b")
plt.plot([4,4,4],t_1200_4,c="b")

for i in range(0,3):
    plt.plot(6,t_1200_6[i],marker=marker_list[i], c="b")
plt.plot([6,6,6],t_1200_6, c="b")

for i in range(0,3):
    plt.plot(8,t_1200_8[i],marker=marker_list[i], c="b")
plt.plot([8,8,8],t_1200_8, c="b") """

plt.ylabel("Deposition thickness [mm]")
plt.ylim((0,4.))
plt.grid(True)  # Enable the grid

exp_t_1200_4 =[2.,2.1,2.22]
exp_t_1200_6 =[1.68,1.75,1.83]
exp_t_1200_8 =[1.6,1.7,1.76]

exp_t_600_6 =[0.,3.6,0.]
exp_t_900_6 =[2.2,2.32,2.45]
exp_t_1500_6 =[1.33,1.396,1.46]

plt.rcParams.update({'font.size': 15})

rpm=1
if rpm == 0:
    plt.errorbar([4,6,8],[t_1200_4[1],t_1200_6[1],t_1200_8[1]], yerr=[[t_1200_4[1]-t_1200_4[0], t_1200_6[1]-t_1200_6[0], t_1200_8[1]-t_1200_8[0]], [t_1200_4[2]-t_1200_4[1], t_1200_6[2]-t_1200_6[1], t_1200_8[2]-t_1200_8[1]]],marker="s",c="b", linestyle="--",label="Simulation", linewidth=2)
    plt.errorbar([4,6,8],[exp_t_1200_4[1],exp_t_1200_6[1], exp_t_1200_8[1] ],  yerr=[[exp_t_1200_4[1]-exp_t_1200_4[0], exp_t_1200_6[1]-exp_t_1200_6[0], exp_t_1200_8[1]-exp_t_1200_8[0]], [exp_t_1200_4[2]-exp_t_1200_4[1], exp_t_1200_6[2]-exp_t_1200_6[1], exp_t_1200_8[2]-exp_t_1200_8[1]]], c="r", marker="^", linestyle= "dotted",label="Experiment", linewidth=2)
    plt.ylim((0,4))
    plt.xlabel("Substrate traverse speed [mm/s]")
    plt.legend()
    #plt.savefig("./deposition/output_plt/Figure_13b.pdf",dpi=300, bbox_inches='tight')
    plt.savefig("./deposition/output_plt/Figure_13b.jpg",dpi=300, bbox_inches='tight')

else:
    plt.errorbar([900,1200,1500], [t_0900_6[1],t_1200_6[1],t_1500_6[1]], yerr=[[t_0900_6[1]-t_0900_6[0], t_1200_6[1]-t_1200_6[0], t_1500_6[1]-t_1500_6[0]], [t_0900_6[2]-t_0900_6[1], t_1200_6[2]-t_1200_6[1], t_1500_6[2]-t_1500_6[1]]],marker="s",c="b", linestyle="--",label="Simulation", linewidth=2)
   # plt.errorbar([600,900,1200,1500], [t_0600_6[1],t_0900_6[1],t_1200_6[1],t_1500_6[1]], yerr=[[t_0600_6[1]-t_0600_6[0], t_0900_6[1]-t_0900_6[0], t_1200_6[1]-t_1200_6[0], t_1500_6[1]-t_1500_6[0]], [t_0600_6[2]-t_0600_6[1], t_0900_6[2]-t_0900_6[1], t_1200_6[2]-t_1200_6[1], t_1500_6[2]-t_1500_6[1]]],marker="s",c="b", linestyle="--",label="Simulation", linewidth=2)
    #plt.errorbar([600,1200], [t_0600_6[1],t_1200_6[1]], yerr=[[t_0600_6[1]-t_0600_6[0], t_1200_6[1]-t_1200_6[0]], [t_0600_6[2]-t_0600_6[1], t_1200_6[2]-t_1200_6[1]]], marker="s", c="b", linestyle="--", label="Simulation", linewidth=2)
    #plt.errorbar([900,1200,1500],[exp_t_900_6[1],exp_t_1200_6[1], exp_t_1500_6[1]], yerr=[[exp_t_900_6[1]-exp_t_900_6[0], exp_t_1200_6[1]-exp_t_1200_6[0], exp_t_1500_6[1]-exp_t_1500_6[0]], [exp_t_900_6[2]-exp_t_900_6[1], exp_t_1200_6[2]-exp_t_1200_6[1], exp_t_1500_6[2]-exp_t_1500_6[1]]], c="r", marker="^", linestyle= "dotted",label="Experiment", linewidth=2)
    #plt.errorbar([600,900,1200,1500],[exp_t_600_6[1],exp_t_900_6[1],exp_t_1200_6[1], exp_t_1500_6[1]], yerr=[[exp_t_600_6[1]-exp_t_600_6[0],exp_t_900_6[1]-exp_t_900_6[0], exp_t_1200_6[1]-exp_t_1200_6[0], exp_t_1500_6[1]-exp_t_1500_6[0]], [exp_t_600_6[2]-exp_t_600_6[1],exp_t_900_6[2]-exp_t_900_6[1], exp_t_1200_6[2]-exp_t_1200_6[1], exp_t_1500_6[2]-exp_t_1500_6[1]]], c="r", marker="^", linestyle= "dotted",label="Experiment", linewidth=2)
    plt.errorbar([900,1200,1500],[exp_t_900_6[1],exp_t_1200_6[1], exp_t_1500_6[1]], yerr=[[exp_t_900_6[1]-exp_t_900_6[0], exp_t_1200_6[1]-exp_t_1200_6[0], exp_t_1500_6[1]-exp_t_1500_6[0]], [exp_t_900_6[2]-exp_t_900_6[1], exp_t_1200_6[2]-exp_t_1200_6[1], exp_t_1500_6[2]-exp_t_1500_6[1]]], c="r", marker="^", linestyle= "dotted",label="Experiment", linewidth=2)
    plt.ylim((0,4))
    plt.xlim((800,1600))
    plt.xlabel("Rod rotational speed [rpm]")
    plt.legend()
    #plt.title("Results of the new model for 1500 rpm only")
    #plt.savefig("./deposition/output_plt/Figure_13d.pdf",dpi=300, bbox_inches='tight')
    plt.savefig("./deposition/output_plt/Figure_13d.jpg",dpi=300, bbox_inches='tight')

plt.show()