import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy import stats as st

# Define the file paths
paths = [
    "./reaction_forces/forces_rod.csv",
    "./reaction_forces/forces_rod_hc.csv",
    "./reaction_forces/forces_rod_lf.csv",
]

# Iterate over each file path
for path in paths:
    print("************************")
    print(f"Processing {path[-14:]}")
    df = pd.read_csv(path)


    # Extract coordinates
    step = df["step"].to_numpy()
    x_force = df["Points:0"].to_numpy()
    y_force = df["Points:1"].to_numpy()
    df["Points:2"] *=-.001
    z_force = df["Points:2"].to_numpy()
    print(np.average(z_force[5:]))

    plt.plot(step,z_force)
    plt.plot([0,2e7],[8,8])
    plt.show()
