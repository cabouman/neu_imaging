import os
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')
import numpy as np

from thomas_process import generate_process, fit_meanshift_2d, fit_meanshift_3d

np.set_printoptions(precision=2, suppress=False)

def fit_score(parent_points, labels, centers):
    distance_score = 0

    # Find the closest cluster center to each parent point
    for parent in parent_points:
        distances = np.linalg.norm(centers - parent, axis=1)
        closest_center_index = np.argmin(distances)
        distance_score += distances[closest_center_index]

    # Penalize for number of clusters
    count_score = (len(np.unique(labels)) - len(parent_points)) ** 2

    print(f"Distance Score: {distance_score}, Count Score: {count_score}")
    return distance_score + count_score

if __name__ == "__main__":
    parent_intensity = 20.0
    parent_time_scale = 100.0
    child_intensity = 300.0
    child_time_scale = 20.0
    child_spread = 1.0

    num_trials = 10

    bandwidths_2d = np.linspace(1, 3, 101)

    bandwidths_3d = np.logspace(0, 1, 101)
    time_scales = np.logspace(1, 2, 101)

    scores_2d = np.zeros((num_trials, len(bandwidths_2d),))
    scores_3d = np.zeros((num_trials, len(bandwidths_3d), len(time_scales)))

    for trial in range(num_trials):
        print(f"Starting Trial {trial+1}/{num_trials}")
        timestamp = np.datetime64('now').astype(str).replace(':', '-').replace(' ', '_')
        test_dir = os.path.join('thomas_process_testing', timestamp)
        os.makedirs(test_dir)

        parent_points, child_points = generate_process(parent_intensity, parent_time_scale,
                                                       child_intensity, child_time_scale, child_spread)

        # Save points (with geometric parameters) to timestamped file for later analysis
        np.savetxt(f"{test_dir}/trial{trial+1}_parent_points.csv", parent_points, delimiter=',', header='time,x,y', 
                   comments=f'parent_intensity={parent_intensity},parent_time_scale={parent_time_scale},child_intensity={child_intensity},child_time_scale={child_time_scale},child_spread={child_spread}\n')
        np.savetxt(f"{test_dir}/trial{trial+1}_child_points.csv", child_points, delimiter=',', header='parent_id,time,x,y',
                   comments=f'parent_intensity={parent_intensity},parent_time_scale={parent_time_scale},child_intensity={child_intensity},child_time_scale={child_time_scale},child_spread={child_spread}\n', fmt='%d,%f,%f,%f')

        for i, bandwidth in enumerate(bandwidths_2d):
            print(f"Trial {trial+1}, 2D Bandwidth {bandwidth}")
            labels_2d, centers_2d = fit_meanshift_2d(child_points, bandwidth=bandwidth)
            scores_2d[trial, i] = fit_score(parent_points[:,1:], labels_2d, centers_2d)

        for i, bandwidth in enumerate(bandwidths_3d):
            for j, time_scale in enumerate(time_scales):
                print(f"Trial {trial+1}, 3D Bandwidth {bandwidth}, Time Scale {time_scale}")
                labels_3d, centers_3d = fit_meanshift_3d(child_points, bandwidth=bandwidth, time_scale=time_scale)
                scores_3d[trial, i, j] = fit_score(parent_points, labels_3d, centers_3d)

        # Save scores (with bandwidth/time_scale/num_trials) to timestamped file for later analysis
        np.savetxt(f"{test_dir}/trial{trial+1}_scores_2d.csv", scores_2d[trial], delimiter=',', header='bandwidth,score')
        np.savetxt(f"{test_dir}/trial{trial+1}_scores_3d.csv", scores_3d[trial], delimiter=',', header='bandwidth,time_scale,score')

    print("2D Scores:", scores_2d)
    print("3D Scores:", scores_3d)

    plt.figure()
    plt.plot(bandwidths_2d, np.log10(scores_2d.mean(axis=0)), marker='o')
    plt.xlabel('Bandwidth')
    plt.ylabel('Average Fit Score (2D)')
    plt.title('2D Mean Shift Fit Score vs Bandwidth')
    plt.grid()

    plt.figure()
    plt.imshow(np.log10(scores_3d.mean(axis=0)), aspect='auto', origin='lower')
    plt.colorbar(label='Average Fit Score (3D)')
    plt.xlabel('Time Scale')
    plt.xticks(ticks=range(len(time_scales)), labels=[f"{ts:.1f}" for ts in time_scales])
    plt.ylabel('Bandwidth')
    plt.yticks(ticks=range(len(bandwidths_3d)), labels=[f"{bw:.1f}" for bw in bandwidths_3d])
    plt.title('3D Mean Shift Fit Score')

    plt.show()