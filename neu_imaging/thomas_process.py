import matplotlib
from matplotlib import pyplot as plt
import sklearn.cluster
matplotlib.use('TkAgg')
import numpy as np

space_size = 50.0  # Size of the spatial domain
padding = 2.0  # Padding around the spatial domain

def generate_parent_points(parent_intensity, parent_time_scale):
    N = np.random.poisson(lam=parent_intensity)
    t = np.random.exponential(scale=parent_time_scale, size=N)
    x = np.random.uniform(0, space_size, size=N)
    y = np.random.uniform(0, space_size, size=N)
    return list(zip(t, x, y))

def generate_child_points(parent_points, child_intensity, child_time_scale, child_spread):
    child_points = []
    for parent_index, (t_parent, x_parent, y_parent) in enumerate(parent_points):
        N_children = np.random.poisson(lam=child_intensity)
        t_children = t_parent + np.random.exponential(scale=child_time_scale, size=N_children)
        x_children = x_parent + np.random.normal(scale=child_spread, size=N_children)
        y_children = y_parent + np.random.normal(scale=child_spread, size=N_children)
        child_points.extend(zip([parent_index] * len(t_children), t_children, x_children, y_children))
    return child_points

def generate_process(parent_intensity, parent_time_scale,
                     child_intensity, child_time_scale, child_spread):
    parent_points = generate_parent_points(parent_intensity, parent_time_scale)
    child_points = generate_child_points(parent_points, child_intensity, child_time_scale, child_spread)
    return np.array(parent_points), np.array(child_points)

def fit_meanshift_2d(points, bandwidth):
    points_2d = points[:,2:]  # Use only spatial dimensions for 2D plot
    fit = sklearn.cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(points_2d)
    labels = fit.labels_
    centers = fit.cluster_centers_
    return labels, centers

def fit_meanshift_3d(points, bandwidth, time_scale):
    scaled_points = points[:,1:].copy()  # Use spatial and time dimensions
    scaled_points[:,0] /= time_scale  # Scale time dimension for better clustering
    fit = sklearn.cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(scaled_points)
    labels = fit.labels_
    centers = fit.cluster_centers_
    centers[:,0] *= time_scale  # Rescale time dimension back to original

    # # Assign time coordinate of centers to minimum time of assigned points
    # for label in np.unique(labels):
    #     centers[label, 0] = np.min(points[labels == label, 1])

    return labels, centers

def plot_process_2d(parent_points, child_points, labels, centers):
    fig = plt.figure()
    plt.scatter(child_points[:,2], child_points[:,3], c=labels, cmap='viridis', alpha=0.5)
    for parent in parent_points:
        plt.scatter(parent[1], parent[2], c='black', s=50, marker='o')
    for center in centers:
        plt.scatter(center[0], center[1], c='red', s=100, marker='X')
    plt.xlim(-padding, space_size + padding)
    plt.ylim(-padding, space_size + padding)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Spatial-Temporal Point Process')

def plot_process_3d(parent_points, child_points, labels, centers):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    ax.scatter(child_points[:,2], child_points[:,3], child_points[:,1], c=labels, cmap='viridis', alpha=0.5)
    for parent in parent_points:
        ax.scatter(parent[1], parent[2], parent[0], c='black', s=50, marker='o') # type: ignore
    for center in centers:
        ax.scatter(center[1], center[2], center[0], c='red', s=100, marker='X', zorder=5) # type: ignore
    ax.set_xlim(-padding, space_size + padding)
    ax.set_ylim(-padding, space_size + padding)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Time') # type: ignore
    plt.title('Spatial-Temporal Point Process')

if __name__ == "__main__":
    parent_points, child_points = generate_process(parent_intensity=50.0, parent_time_scale=100.0,
                                 child_intensity=300.0, child_time_scale=20.0, child_spread=1.0)
    labels_2d, centers_2d = fit_meanshift_2d(child_points, 1.5)
    labels_3d, centers_3d = fit_meanshift_3d(child_points, 2.5, 25)
    plot_process_2d(parent_points, child_points, labels_2d, centers_2d)
    plot_process_3d(parent_points, child_points, labels_3d, centers_3d)
    plt.show()