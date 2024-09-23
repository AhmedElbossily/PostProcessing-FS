
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def smooth_temperatures(temperatures, window_size):
  """
  Smooths a list of temperatures using a moving average.

  Args:
      temperatures: A list of temperature values.
      window_size: The number of neighboring temperatures to consider in the average.

  Returns:
      A new list containing the smoothed temperatures.
  """
  smoothed_temps = np.convolve(temperatures, np.ones(window_size), mode='same') / window_size
  return smoothed_temps.tolist()


def get_filenames(path):
    return sorted([f for f in os.listdir(path)])

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

global_temp_list_each_step=[]
def function(files_list, path_1, ex_x, ex_y, avarage_error_list,max_temp_error,chart_tilte):
   #files_list=files_list[int(len(files_list)/2):]
   
   for idx, file in enumerate(files_list):
     path = path_1+"/"+ file
     df = pd.read_csv(path)
     df["Points:0"] *= 1000
     df["Points:1"] *= 1000
     df["Points:2"] *= 1000
     df["Temperature"] -= 273
     df = df.drop(df[df['Points:2'] < 0.5].index)
     start = -40
     stop = 40  # Note: stop is exclusive in rang
     step = 2
     # Use vectorized operations for efficiency
     bins = np.arange(start, stop + step, step)  # Create bins directly with numpy
     digitized = np.digitize(df["Points:1"], bins)  # Assign data points to bins

     # Group by bin and find max temperature efficiently
     temp_list = df.groupby(digitized)["Temperature"].max().to_numpy()
     temp_list = smooth_temperatures(temp_list, 2)
     temp_list[0] = temp_list[1]
     #temp_list[-1] = temp_list[-2]

     if len(files_list) >1:
       global_temp_list_each_step.append(temp_list)
      
   return bins, temp_list

def plot_chart(bins, temp_list, ex_x, ex_y,chart_tilte):
   # setting font sizeto 30
   plt.rcParams.update({'font.size': 12})
   plt.plot(bins, temp_list, label="Simulation", color = "b")  
   plt.scatter(ex_x, ex_y, label="Experiment",marker="*", color="r")
   plt.text(-39, 215, 'Advancing side')
   plt.text(15, 215, 'Retreating side')

   plt.title(chart_tilte)
   plt.xlabel("Distance to substrate center [mm]")
   plt.ylabel("Temperature [°C]")
   plt.legend()
   plt.grid()
   plt.savefig("./output/"+chart_tilte[:3]+chart_tilte[4]+".pdf")
   plt.show() 

####################################################################################################################################################     
chart_tilte=""
#path_1 = './1200-6'  #set A
#path_1 = './900-6'  #set B # change smoothing length to 3
path_1 = './1500-6' #set C
#path_1 = './1200-4' #set D
#path_1 = './1200-8' #set E

if path_1 =='./1200-6':
  chart_tilte = "Set A: 1200 rpm, 6 mm/s"

if path_1 =='./900-6':
  chart_tilte = "Set B: 900 rpm, 6 mm/s"

if path_1 =='./1500-6':
  chart_tilte = "Set C: 1500 rpm, 6 mm/s"

if path_1 =='./1200-4':
  chart_tilte = "Set D: 1200 rpm, 4 mm/s"

if path_1 =='./1200-8':
  chart_tilte = "Set E: 1200 rpm, 8 mm/s"

files_list = get_sorted_files (path_1)
avarage_error_list =[]
max_temp_error =[]

path_ex= "./ex/ex_"+path_1[2:]+".csv"
df_ex = pd.read_csv(path_ex)
ex_x =df_ex["x"].to_numpy()
ex_y = df_ex["y"].to_numpy()

bins, temp_list = function(files_list, path_1, ex_x, ex_y, avarage_error_list,max_temp_error, chart_tilte)

max_temp_values = [max(column) for column in zip(*global_temp_list_each_step)]

interp_func = interp1d(bins, max_temp_values, kind='linear')  # Choose linear interpolation

# Interpolate temperatures at ex_x values
interp_temp = interp_func(ex_x)
error = []
for i, val in enumerate(ex_x):
  e =100.*(abs(ex_y[i]- interp_temp[i])/ex_y[i])
  error.append(e)
  print(e)

print("av error: ", np.mean(error))
plot_chart(bins, max_temp_values, ex_x, ex_y, chart_tilte)
