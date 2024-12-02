import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy import stats as st

# Define the file paths
""" paths = [
    "./reaction_forces_1200/forces_rod_0ffl.csv",
    "./reaction_forces_1200/forces_rod_1ffl.csv",
    "./reaction_forces_1200/forces_rod_3ffl.csv",
] """

paths = [
    "./reaction_forces_1500/forces_rod_0ffl.csv",
    "./reaction_forces_1500/forces_rod_1ffl.csv",
    "./reaction_forces_1500/forces_rod_3ffl.csv",
] 


# Define the line styles
line_styles = ['--', '-', ':']

# Create a figure and axis
plt.figure()

simulated = ["1st","2nd","3rd"]
avg_z_force = []

# Iterate over each file path
for i, path in enumerate(paths):
    print("************************")
    print(f"Processing {path[-14:]}")
    df = pd.read_csv(path)

    # Extract coordinates
    step = df["step"].to_numpy()
    x_force = df["Points:0"].to_numpy()
    y_force = df["Points:1"].to_numpy()
    df["Points:2"] *= -.001
    z_force = df["Points:2"].to_numpy()
    print(np.average(z_force[5:]))
    avg_z_force.append(np.average(z_force[5:]))

    # Plot the data with specified line style
    plt.plot(step, z_force, line_styles[i], label=simulated[i])

# Add a horizontal line at y=8
plt.plot([0, 2e7], [8, 8], 'k--', label="exp = 8 N")

# Add labels and legend
plt.xlabel('Step')
plt.ylabel('Reaction Force')
plt.legend()
plt.show()
# Create a new figure for the average plot
plt.figure()


    
    # Plot the average with specified line style
plt.bar(simulated, avg_z_force)


# Add labels and legend
plt.xlabel('Simulation')
plt.ylabel('Average Reaction Force')
plt.legend()
plt.show()