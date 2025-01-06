
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

def function(files_list, path_1, ex_x, ex_y, avarage_error_list,max_temp_error,chart_tilte):
   for idx, file in enumerate(files_list):
     path = path_1+"/"+ file
     df = pd.read_csv(path)
     df["Points:0"] *= 1000
     df["Points:1"] *= 1000
     df["Points:2"] *= 1000
     df["Temperature"] -= 273
     #df = df.drop(df[df['Points:2'] < -.5].index)
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

     # Create interpolation function from bins and temp_list
     interp_func = interp1d(bins, temp_list, kind='linear')  # Choose linear interpolation

     # Interpolate temperatures at ex_x values
     interp_temp = interp_func(ex_x)
     error = []
     for i, val in enumerate(ex_x):
      e =100.*(abs(ex_y[i]- interp_temp[i])/ex_y[i])
      error.append(e)
      #print(f"x: {val:.2f}, Experimental Temperature: {ex_y[i]:.2f}, Interpolated Temperature: {interp_temp[i]:.2f}, Error:{e:.2f}")
     
     avarage_error = np.mean(error)
     avarage_error_list.append(avarage_error)
     max_temp_error.append(error[3])

     if len(files_list) ==1:
       print("------")
       print("Errors")
       print(' & ',error[0],' & ',error[1],' & ' ,error[2], ' & ',error[3],' & ' ,error[4],' & ' ,error[5], ' & ',error[6], ' & ',error[7], ' & ',avarage_error)
       
       text_file = open("./output/"+chart_tilte[:3]+chart_tilte[4]+".txt", "w")

       text_file.write(' & ' + str(round(error[0],1)) +
                       ' & ' + str(round(error[1],1)) +
                       ' & ' + str(round(error[2],1)) +
                       ' & ' + str(round(error[3],1)) +
                       ' & ' + str(round(error[4],1)) +
                       ' & ' + str(round(error[5],1)) +
                       ' & ' + str(round(error[6],1)) +
                       ' & ' + str(round(error[7],1)) +
                       ' & '+ str(round(avarage_error,1)))
       text_file.close()

   return bins, temp_list

def plot_chart(bins, temp_list, ex_x, ex_y,chart_tilte):
   # setting font sizeto 30
   plt.rcParams.update({'font.size': 15})
   plt.plot(bins, temp_list, label="Simulation", color = "b")  
   #plt.scatter(ex_x, ex_y, label="Experiment",marker="*", color="r")
   plt.text(-39, 215, 'Advancing side')
   plt.text(15, 215, 'Retreating side')

   #plt.title(chart_tilte)
   plt.xlabel("Distance to substrate center [mm]")
   plt.ylabel("Temperature [°C]")
   plt.legend()
   plt.grid()
   plt.savefig("./output/"+chart_tilte[:3]+chart_tilte[4]+".pdf")
   plt.savefig("./output/"+chart_tilte[:3]+chart_tilte[4]+".png")
   plt.show() 

####################################################################################################################################################     
chart_tilte=""
#path_1 = './1200-6'  #set A
#path_1 = './900-6'  #set B # change smoothing length to 3
#path_1 = './1500-6' #set C
#path_1 = './1200-4' #set D
path_1 = './1200-8' #set E

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
ex_x =df_ex["x"].to_numpy()-2.5
ex_y = df_ex["y"].to_numpy()

function(files_list, path_1, ex_x, ex_y, avarage_error_list,max_temp_error, chart_tilte)


print("--------------------------------------------------")
print("Selecting time step based on minimum avarage error")
min_average = np.min(avarage_error_list)
index_min_average = avarage_error_list.index(min_average)
file_name_min_average=files_list[index_min_average]
print("min_average:", min_average)
print("index min_average:", index_min_average)
print("max_temp_error:", max_temp_error[index_min_average])
print("min_average file name:", file_name_min_average)

print("-------------------------------------------")
print("Selecting time step based on max temp error")
min_max_temp_error = np.min(max_temp_error)
index_max_temp_error = max_temp_error.index(min_max_temp_error)
print("min_max_temp_error:", min_max_temp_error)
print("index_max_temp_error:", index_max_temp_error)
print("avarage error:", avarage_error_list[index_max_temp_error])
print("min_average file name:", files_list[index_max_temp_error])

files_list = []
avarage_error_list =[]
max_temp_error =[]
bins =[]
temp_list= []

files_list.append(file_name_min_average)
bins,temp_list = function(files_list, path_1, ex_x, ex_y, avarage_error_list,max_temp_error, chart_tilte)
plot_chart(bins, temp_list, ex_x, ex_y, chart_tilte)

