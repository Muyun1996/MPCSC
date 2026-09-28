import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_data(offline_obs, online_obs):

    combined_data = np.vstack((offline_obs, online_obs))

    tsne = TSNE(n_components=2, random_state=42)
    reduced_data = tsne.fit_transform(combined_data)

    reduced_array1 = reduced_data[:offline_obs.shape[0]]
    reduced_array2 = reduced_data[offline_obs.shape[0]:]

    sns.set(style='whitegrid')

    plt.scatter(reduced_array1[:, 0], reduced_array1[:, 1], c='red', label='offline_obs', alpha=0.05)
    plt.scatter(reduced_array2[:, 0], reduced_array2[:, 1], c='blue', label='online_obs', alpha=0.05)

    plt.legend()
    plt.title('2D Visualization of High-Dimensional Points')

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiments_dir",
        type=str,
        default=None,
        help="The directory where the original experiment was run.",
    )

    args = parser.parse_args()
    args.experiment_dir = "/Users/moumuyun/mbrl-lib/mbrl/examples/exp/pets/default/gym___gym_dynamics/Boids_dynamics/2023.05.05/143144"

    work_dir = args.experiment_dir
    replay_buffer_offline = np.load(work_dir + "/replay_buffer_offline.npz")
    replay_buffer_online = np.load(work_dir + "/replay_buffer_online.npz")
    replay_buffer_test = np.load(work_dir + "/replay_buffer_test.npz")

    offline_obs = replay_buffer_offline['obs']
    online_obs = replay_buffer_online['obs']

    print("offline_obs.shape = ", offline_obs.shape)
    print("online_obs.shape = ", online_obs.shape)


    visualize_data(offline_obs, online_obs)







