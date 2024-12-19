import csv
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt



variable = '0900'
def read_3d_points(filepath):
    df = pd.read_csv(filepath)
    df = df.applymap(lambda x: float(x) * 1000)
    df.iloc[:, 0] = df.iloc[:, 0] + 60
    df.iloc[:, 1] = df.iloc[:, 1] * -1
    df = df[df.iloc[:, 0] >= 40]
    df = df[df.iloc[:, 0] <= 50]
    points = df.values

    return points

def calculate_length_width(points):
    points = np.array(points)
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    length = np.max(x_coords) - np.min(x_coords)
    width = np.max(y_coords) - np.min(y_coords)
    return length, width

def sample_width(points_total, points, step):
    points = np.array(points)
    y_coords = points[:, 1]
    points_total = np.array(points_total)
    y_coords_total = points_total[:, 1]
    """ min_y = np.min(y_coords_total)
    max_y = np.max(y_coords_total) """
    min_y = -8
    max_y = 10   
    bins = np.arange(min_y, max_y + step, step)
    counts, _ = np.histogram(y_coords, bins=bins)
    return bins, counts 


def plot_total_points(points_total, title, bins):
    x_values_max = np.full(len(bins), 70)
    x_values_min = np.full(len(bins), 0)
    plt.figure(figsize=(12, 6))
    plt.scatter(points_total[:, 0], points_total[:, 1], s=400, c='blue', alpha=1, label='deposited particles') 
    for i in range(0, len(bins)):
        if i == 0:
            plt.plot([x_values_min[i], x_values_max[i]], [bins[i], bins[i]], color='lightgreen', label='sample steps')
        else:
            plt.plot([x_values_min[i], x_values_max[i]], [bins[i], bins[i]], color='lightgreen')
    plt.xlabel('X coordinates')
    plt.ylabel('Y coordinates')
    #plt.legend()
    plt.savefig(f'./fig/{title}_total_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_data(filepaths, step=2):
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 20})
    plt.xlim((-11, 14))

    colors = ['blue', 'red']
    lb = ['Outer particles', 'Inner particles']
    for i, (filepath, filepath_total) in enumerate(filepaths):
        points = read_3d_points(filepath)
        points_total = read_3d_points(filepath_total)
        bins, counts = sample_width(points_total, points, step)
        
        bin_centers = (bins[:-1] + bins[1:]) / 2
        plt.bar(bins[:-1], (counts/len(points))*100., width=np.diff(bins), edgecolor='black', align='edge', alpha=0.75, label=lb[i], color=colors[i])
        #plt.scatter(bin_centers, counts, alpha=0.75, color=colors[i])
        
        spl = make_interp_spline(bin_centers, counts, k=2)
        smooth_bins = np.linspace(bin_centers.min(), bin_centers.max(), 100)
        smooth_counts = spl(smooth_bins)
        #plt.plot(smooth_bins, smooth_counts, alpha=1, color="black")

        #if i == 1:
           # plot_total_points(points_total, f'Total Points {variable}_{i}', bins)
        
    #plt.legend(loc='upper left')
    plt.savefig(f'./fig/{variable}.png', dpi=300, bbox_inches='tight')
    plt.show()

# Example usage
filepaths = [
    #(f'./out_{variable}_r3.csv', f'./total_{variable}.csv'),
    (f'./in_{variable}_r3.csv', f'./total_{variable}.csv'),
]

plot_data(filepaths)