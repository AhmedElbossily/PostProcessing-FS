
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
     df = df.drop(df[df['Points:2'] < -.5].index)
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

paths = [
  "./deposition/1200-4_0.75.csv",
  "./deposition/1200-6_0.75.csv",
  "./deposition/1200-8_0.75.csv",
  "./deposition/1500-6_0.75.csv",
  "./deposition/0900-6_0.75.csv",
]
points_data=[]
for path in paths:
  p_size = float(path[-8:-4])
  print("p_size: ", p_size)
  df = pd.read_csv(path)
  df["Points:0"] *= 1000
  df["Points:0"] += 60
  df["Points:1"] *= -1000
  df["Points:2"] *= 1000

  start = 35 # Start from 35 mm
  stop = 53   # Stop at 53 mm
  step = 2

  # Use vectorized operations for efficiency
  bins = np.arange(start, stop + step, step)  # Create bins directly with numpy
  digitized = np.digitize(df["Points:0"], bins)  # Assign data points to bins

  # Group by bin and find max temperature efficiently
  y_max_list = df.groupby(digitized)["Points:1"].max().to_numpy()
  y_max_list = y_max_list[1:]
  y_max_list += p_size / 2.
  y_min_list = df.groupby(digitized)["Points:1"].min().to_numpy()
  y_min_list = y_min_list[1:]
  y_min_list -= p_size / 2.
  print(len(bins))
  print(len(y_max_list))
  print(len(y_min_list))

  plt.rcParams["figure.figsize"] = (13, 5)
  plt.rcParams.update({'font.size': 12})
  plt.text(13, 7, 'Advancing side')
  plt.text(13, -7, 'Retreating side')
  plt.text(7, 1, 'Deposition')
  plt.text(7, -2, 'Direction')
  plt.scatter(df["Points:0"].to_numpy(), df["Points:1"].to_numpy(), marker="o", color="royalblue", s=300)

  for i in range(0, len(bins)):
    if i == 0:
      plt.plot([bins[i], bins[i]], [y_max_list[i], y_min_list[i]], linewidth=2, color="lightgreen",
           label="Sampled width")
    else:
      plt.plot([bins[i], bins[i]], [y_max_list[i], y_min_list[i]], linewidth=2, color="lightgreen")

  width = []
  centerline = []
  for i in range(len(y_max_list)):
    width.append(y_max_list[i] - y_min_list[i])
    centerline.append((y_max_list[i] + y_min_list[i])/2.)

  point_data = []
  point_data.append(np.min(width))
  point_data.append(np.average(width))
  point_data.append(np.max(width))

  points_data.append(point_data)

  print(f"Deposition width Distribution: {path[-15:-4]}")
  print("min Width", np.min(width))
  print("Average Width", np.average(width))
  print("max Width", np.max(width))

  plt.plot(bins, y_max_list, color="r", linewidth=2, label="Deposition edges")
  plt.plot(bins, y_min_list, color="r", linewidth=2)
  plt.plot(bins, centerline, color="r", linewidth=2)
  plt.arrow(x=7, y=0, dx=55, dy=0, width=.2, color="black")
  plt.xlabel("Deposition length, x direction [mm]")
  plt.ylabel("Deposition width, y direction [mm]")

  plt.legend()
  plt.title(f"Deposition width Distribution: {path[-15:-4]}")


  plt.savefig(f"./deposition/output_plt_width/width_validation_{path[-15:-4]}.pdf")
  plt.show()

plt.rcParams.update({'font.size': 15})
plt.ylim((0,35))
plt.grid(True)  # Enable the grid
plt.ylabel("Deposition width [mm]")



rpm= 1
if rpm == 0:
    plt.errorbar([4,6,8],[points_data[0][1],points_data[1][1], points_data[2][1]], yerr= [[points_data[0][1]-points_data[0][0], points_data[1][1]-points_data[1][0], points_data[2][1]-points_data[2][0]],[points_data[0][2]-points_data[0][1], points_data[1][2]-points_data[1][1], points_data[2][2]-points_data[2][1]],],marker="s",c="b", linestyle="--",label="Simulation", linewidth=2)
    exp = [[18.3,19.1 ,19.8],
           [17.5, 18.3,18.9 ],
           [17., 17.7, 18.6]]
    plt.errorbar([4,6,8],[exp[0][1],exp[1][1], exp[2][1]], yerr= [[exp[0][1]-exp[0][0], exp[1][1]-exp[1][0], exp[2][1]-exp[2][0]],[exp[0][2]-exp[0][1], exp[1][2]-exp[1][1], exp[2][2]-exp[2][1]]], c="r", marker="^", linestyle= "dotted",label="Experiment", linewidth=2)
    plt.xlabel("Substrate traverse speed [mm/s]")
    plt.legend()
    plt.savefig("./deposition/output_plt_width/wv_ts.pdf")

else:
    plt.errorbar([900,1200,1500],[points_data[4][1],points_data[1][1], points_data[3][1]], yerr= [[points_data[4][1]-points_data[4][0], points_data[1][1]-points_data[1][0], points_data[3][1]-points_data[3][0]],[points_data[4][2]-points_data[4][1], points_data[1][2]-points_data[1][1], points_data[3][2]-points_data[3][1]]],marker="s",c="b", linestyle="--",label="Simulation",linewidth=2)
   
    exp = [[18.3, 19.2, 20.1],
           [17.5, 18.3, 18.9],
           [16.3, 16.8, 17.4]]
    plt.errorbar([900,1200,1500],[exp[0][1],exp[1][1], exp[2][1]], yerr= [[exp[0][1]-exp[0][0], exp[1][1]-exp[1][0], exp[2][1]-exp[2][0]],[exp[0][2]-exp[0][1], exp[1][2]-exp[1][1], exp[2][2]-exp[2][1]]], c="r", marker="^", linestyle= "dotted",label="Experiment", linewidth=2)
    plt.xlabel("Rod rotational speed [rpm]")
    plt.legend()
    plt.savefig("./deposition/output_plt_width/wv_rpm.pdf") 

plt.show()