import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d



# Define the file path
file_path ='./material_flow/deposition.csv'  #set A

# Initialize an empty list to store lines
lines = []

# Open the file and read it line by line
with open(file_path, 'r') as file:
    for line in file:
        # Strip any leading/trailing whitespace and add the line to the list
        lines.append(line.strip())

# Join the list into a single string, separated by commas
result = ','.join(lines)

# Print the final string
print(result)
