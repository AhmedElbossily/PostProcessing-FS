import csv
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt



variable = '1200'

def read_3d_points(filepath):
    df = pd.read_csv(filepath)
    df = df.applymap(lambda x: float(x) * 1000)
    df.iloc[:, 0] = df.iloc[:, 0] + 60
    df.iloc[:, 1] = df.iloc[:, 1] * -1
    df = df[df.iloc[:, 0] >= 40]
    df = df[df.iloc[:, 0] <= 50]
    points = df.values

    return points


def sample_width(points_total, points, step):
    points = np.array(points)
    y_coords = points[:, 1]
    points_total = np.array(points_total)
    y_coords_total = points_total[:, 1]
    """ min_y = np.min(y_coords_total)
    max_y = np.max(y_coords_total) """
    min_y = -8
    max_y = 12
    bins = np.arange(min_y, max_y + step, step)
    counts, _ = np.histogram(y_coords, bins=bins)
    return bins, counts
 
def plot_total_points(points_total, title, bins):
    x_values_max = np.full(len(bins), 50)
    x_values_min = np.full(len(bins), 40)
    #plt.figure(figsize=(13, 5))
    plt.scatter(points_total[:, 0], points_total[:, 1], s=400, c='royalblue', alpha=1, label='deposited particles') 
    for i in range(0, len(bins)):
        if i == 0:
            plt.plot([x_values_min[i], x_values_max[i]], [bins[i], bins[i]], color='lightgreen', label='sample steps')
        else:
            plt.plot([x_values_min[i], x_values_max[i]], [bins[i], bins[i]], color='lightgreen')

    plt.plot([x_values_min[0], x_values_min[0]], [bins[0], bins[-1]], color='lightgreen')
    plt.plot([x_values_max[0], x_values_max[0]], [bins[0], bins[-1]], color='lightgreen')
    plt.arrow(x=7, y=0, dx=55, dy=0, width=.2, color="black")

    
    """ plt.xlabel('X coordinates')
    plt.ylabel('Y coordinates') """
    #plt.legend()
    plt.savefig(f'./fig/{title}_total_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_data(filepaths, step=2):
    #plt.figure(figsize=(13, 5))
    #plt.figure(figsize=(12, 6))
    plt.rcParams.update({'font.size': 20})
    plt.xlim((-11, 14))

    colors = ['red', 'blue']
    lb = ['Outer particles', 'Inner particles']
    old_counts = []
    for i, (filepath, filepath_total) in enumerate(filepaths):
        points = read_3d_points(filepath)
        points_total = read_3d_points(filepath_total)
        bins, counts = sample_width(points_total, points, step)
        bins_total, counts_total = sample_width(points_total, points_total, step)
        if i == 0:
            plt.bar(bins[:-1], (counts/counts_total)*100., width=np.diff(bins), edgecolor='black', align='edge', alpha=0.75, label=lb[i], color=colors[i])
            old_counts = (counts/counts_total)*100.
        else:
            plt.bar(bins[:-1], (counts/counts_total)*100., bottom= old_counts, width=np.diff(bins), edgecolor='black', align='edge', alpha=0.75, label=lb[i], color=colors[i])
            old_counts += (counts/counts_total)*100.

        #if i == 1:
          # plot_total_points(points_total, f'Total Points {variable}_{i}', bins) 
       
    plt.savefig(f'./fig/{variable}.png', dpi=300, bbox_inches='tight')
    plt.show()

# Example usage
filepaths = [
    (f'./in_{variable}_r3.csv', f'./total_{variable}.csv'),
    (f'./out_{variable}_r3.csv', f'./total_{variable}.csv'),

]

plot_data(filepaths)