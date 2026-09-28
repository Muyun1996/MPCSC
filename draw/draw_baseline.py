import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def draw_baseline(env_name="kuramoto"):
    if env_name == "kuramoto":
        df_no_control = pd.read_csv("data/baseline_kuramoto_no_control.csv", names=['No Control'])
        df_random = pd.read_csv("data/baseline_kuramoto_random.csv", names=['Random'])
        df_ppo = pd.read_csv("data/baseline_kuramoto_ppo.csv", names=['PPO'])
        df_pets = pd.read_csv("data/baseline_kuramoto_pets.csv", names=['PETS'])
        df_ours = pd.read_csv("data/baseline_kuramoto_ours.csv", names=['MPCSC'])
    elif env_name == "boids":
        df_no_control = pd.read_csv("data/baseline_boids_no_control.csv", names=['No Control'])
        df_random = pd.read_csv("data/baseline_boids_random.csv", names=['Random'])
        df_ppo = pd.read_csv("data/baseline_boids_ppo.csv", names=['PPO'])
        df_pets = pd.read_csv("data/baseline_boids_pets.csv", names=['PETS'])
        df_ours = pd.read_csv("data/baseline_boids_ours.csv", names=['MPCSC'])
    elif env_name == "sis":
        df_no_control = pd.read_csv("data/baseline_sis_no_control.csv", names=['No Control'])
        df_random = pd.read_csv("data/baseline_sis_random.csv", names=['Random'])
        df_pets = pd.read_csv("data/baseline_sis_pets.csv", names=['PETS'])
        df_ours = pd.read_csv("data/baseline_sis_ours.csv", names=['MPCSC'])

        df_no_control = 1 - df_no_control * -1
        df_random = 1 - df_random * -1
        df_pets = 1 - df_pets * -1
        df_ours = 1 - df_ours * -1


    if env_name == "kuramoto" or env_name == "boids":
        df = pd.concat([df_no_control, df_random, df_ppo, df_pets, df_ours], axis=1)
        colors = ["black", "green", "red", "gray", "blue"]
        linestyles = ["dashed", "--", ":", "-.", '-']
        labels = ["No Control", "Random", "PPO", "PETS", "MPCSC"]
    elif env_name == "sis":
        df = pd.concat([df_no_control, df_random, df_pets, df_ours], axis=1)
        colors = ["black", "green", "gray", "blue"]
        linestyles = ["dashed", "--", "-.", '-']
        labels = ["No Control", "Random", "PETS", "MPCSC"]

    sns.set(font_scale=1.3)
    sns.set_palette("deep")
    plt.figure(figsize=(6, 5))

    for i, label in enumerate(labels):
        sns.lineplot(data=df, y=label, x=df.index,
                     linewidth=3,
                     color=colors[i],
                     linestyle=linestyles[i],
                     label=label)

    # plt.title('Comparison of Baseline Methods')
    if env_name == "kuramoto":
        plt.legend(loc='upper left')
    elif env_name == "boids":
        plt.legend(loc='lower right')
    plt.xlabel('Time Step')

    if env_name == "kuramoto" or env_name == "boids":
        plt.ylabel('Order Parameter')
    elif env_name == "sis":
        plt.ylabel('Susceptible Population Proportion')

    plt.savefig('results/' + env_name + '_baseline.pdf', format='pdf')
    plt.show()


draw_baseline(env_name="kuramoto")
draw_baseline(env_name="boids")
draw_baseline(env_name="sis")