
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import os


def get_sorted_files(directory):
  """
  Returns a sorted list of all files in the given directory, sorted by the "yy" value 
  in the file name (xxxx-x_yy.csv format).

  Args:
      directory: The path to the directory containing the files.

  Returns:
      A list of file names sorted by the "yy" value.

  Raises:
      OSError: If the directory path is invalid or inaccessible.
  """

  try:
    files = os.listdir(directory)
  except OSError as e:
    raise OSError(f"Error accessing directory: {directory}") from e

  # Filter only files with the expected format
  filtered_files = [f for f in files if f.endswith(".csv") and "_" in f]

  # Extract the "yy" value and use it for sorting (assuming fixed format)
  def get_yy_value(filename):
    return int(filename.split("_")[-1].split(".")[0])

  sorted_files = sorted(filtered_files, key=get_yy_value)
  return sorted_files


path_1 = './nj'
files_list=[]
files_list = get_sorted_files (path_1)
for i in range(0,570):
  temperatures = []
  pl = []
  for idx, file in enumerate(files_list):
    path = path_1+"/"+ file
    df = pd.read_csv(path)
    df["Points:0"] *= 1000
    df["Points:1"] *= 1000
    df["Points:2"] *= 1000
    df["Temperature"] -= 273
    df = df.sort_values(by="unique_idx").reset_index(drop=True)
    temperatures.append(df["Temperature"].iloc[i])
    pl.append(df["Plastic_Strain"].iloc[i])
  print(np.max(pl))
  plt.plot(pl)
  plt.show()

