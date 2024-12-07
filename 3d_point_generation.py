import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ...existing code...

def make_points_denser(points, values, factor=2):
    """
    Generate denser points in the 3D domain by interpolating between existing points.
    
    :param points: numpy array of shape (n, 3) representing the original points
    :param values: numpy array of shape (n,) representing the values (0 or 1) of the original points
    :param factor: int, the factor by which to increase the density of points
    :return: tuple of numpy arrays representing the denser points and their values
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Create a grid of points
    xi = np.linspace(x.min(), x.max(), len(x) * factor)
    yi = np.linspace(y.min(), y.max(), len(y) * factor)
    zi = np.linspace(z.min(), z.max(), len(z) * factor)
    xi, yi, zi = np.meshgrid(xi, yi, zi)

    # Interpolate the points
    dense_points = griddata(points, points, (xi, yi, zi), method='linear')
    dense_values = griddata(points, values, (xi, yi, zi), method='nearest')

    # Reshape the dense_points array to a 2D array
    dense_points = dense_points.reshape(-1, 3)
    dense_values = dense_values.reshape(-1)

    # Remove NaN values resulting from interpolation
    valid_mask = ~np.isnan(dense_points).any(axis=1)
    dense_points = dense_points[valid_mask]
    dense_values = dense_values[valid_mask]

    return dense_points, dense_values

def plot_points(old_points, old_values, new_points, new_values):
    """
    Plot the old and new points in a 3D scatter plot.
    
    :param old_points: numpy array of shape (n, 3) representing the original points
    :param old_values: numpy array of shape (n,) representing the values of the original points
    :param new_points: numpy array of shape (m, 3) representing the denser points
    :param new_values: numpy array of shape (m,) representing the values of the denser points
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot old points
    ax.scatter(old_points[:, 0], old_points[:, 1], old_points[:, 2], c=old_values, marker='o', label='Original Points')
    
    # Plot new points
    ax.scatter(new_points[:, 0], new_points[:, 1], new_points[:, 2], c=new_values, marker='^', label='Denser Points')
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.legend()
    
    plt.show()

# Example usage
if __name__ == "__main__":
    # Example points and values
    points = np.random.rand(100, 3)
    values = np.random.choice([0, 1], size=100)
    denser_points, denser_values = make_points_denser(points, values, factor=2)
    
    # Plot the points
    plot_points(points, values, denser_points, denser_values)
