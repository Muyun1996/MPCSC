import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter

def draw_episode_reward(file_path='kuramoto_episode_reward.csv', nrows=10):
    file_path = file_path
    data = pd.read_csv(file_path, nrows=nrows)
    x_data = data['env_step']

    window_size = 3

    # Extract columns for mean, min, and max time series data
    mean_data15000 = data['algorithm.initial_obs_steps: 15000 - episode_reward']
    min_data15000 = data['algorithm.initial_obs_steps: 15000 - episode_reward__MIN']
    max_data15000 = data['algorithm.initial_obs_steps: 15000 - episode_reward__MAX']
    # Calculate the moving average for each time series data
    mean_data15000 = mean_data15000.rolling(window=window_size).mean()
    min_data15000 = min_data15000.rolling(window=window_size).mean()
    max_data15000 = max_data15000.rolling(window=window_size).mean()

    mean_data1500 = data['algorithm.initial_obs_steps: 1500 - episode_reward']
    min_data1500 = data['algorithm.initial_obs_steps: 1500 - episode_reward__MIN']
    max_data1500 = data['algorithm.initial_obs_steps: 1500 - episode_reward__MAX']
    mean_data1500 = mean_data1500.rolling(window=window_size).mean()
    min_data1500 = min_data1500.rolling(window=window_size).mean()
    max_data1500 = max_data1500.rolling(window=window_size).mean()

    mean_data2 = data['algorithm.initial_obs_steps: 2 - episode_reward']
    min_data2 = data['algorithm.initial_obs_steps: 2 - episode_reward__MIN']
    max_data2 = data['algorithm.initial_obs_steps: 2 - episode_reward__MAX']
    mean_data2 = mean_data2.rolling(window=window_size).mean()
    min_data2 = min_data2.rolling(window=window_size).mean()
    max_data2 = max_data2.rolling(window=window_size).mean()

    # Plot the data from both datasets as line plots with shaded regions
    sns.set(font_scale=1.5)
    custom_palette = ['blue', 'orange', 'green']  # 自定义的颜色调色板，可以根据需要添加更多颜色
    sns.set_palette("deep")

    plt.figure(figsize=(6, 5))

    plt.plot(x_data, mean_data15000, label='15000 obs', linewidth=3, marker='*')
    plt.fill_between(x_data,  min_data15000, max_data15000, alpha=0.13)

    plt.plot(x_data, mean_data1500, label='1500 obs', linewidth=3, marker='o')
    plt.fill_between(x_data,  min_data1500, max_data1500, alpha=0.13)

    plt.plot(x_data, mean_data2, label='0 obs', linewidth=3, marker='s')
    plt.fill_between(x_data,  min_data2, max_data2, alpha=0.13)

    # 使用科学记数法显示 y 轴刻度标签
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #ax.yaxis.offsetText.set_fontsize(12)  # 调整偏移文本大小
    ax.yaxis.get_major_formatter().set_powerlimits((0, 0))  # 控制指数部分的显示范围

    plt.legend(loc='lower right')
    plt.xlabel('Environment Steps')
    plt.ylabel('Episode Return')
    plt.title('')
    plt.tight_layout()

    base_name = file_path.split('/')[-1]  # 提取最后一个 '/' 之后的部分，得到 'kuramoto_episode_reward_init_obs.csv'
    file_name_without_extension = base_name.split('.')[0]  # 提取 '.' 之前的部分
    plt.savefig("results/" + file_name_without_extension + '.pdf', format='pdf')
    plt.show()


def draw_loss(file_path='kuramoto_val_loss.csv', indicator="model_val_score", nrows=10):
    file_path = file_path
    data = pd.read_csv(file_path, nrows=nrows)
    x_data = data['iteration']
    window_size = 1

    # Extract columns for mean, min, and max time series data
    mean_data15000 = data['algorithm.initial_obs_steps: 15000 - ' + indicator]
    min_data15000 = data['algorithm.initial_obs_steps: 15000 - ' + indicator + '__MIN']
    max_data15000 = data['algorithm.initial_obs_steps: 15000 - ' + indicator + '__MAX']
    # Calculate the moving average for each time series data
    mean_data15000 = mean_data15000.rolling(window=window_size).mean()
    min_data15000 = min_data15000.rolling(window=window_size).mean()
    max_data15000 = max_data15000.rolling(window=window_size).mean()

    mean_data1500 = data['algorithm.initial_obs_steps: 1500 - ' + indicator]
    min_data1500 = data['algorithm.initial_obs_steps: 1500 - ' + indicator + '__MIN']
    max_data1500 = data['algorithm.initial_obs_steps: 1500 - ' + indicator + '__MAX']
    mean_data1500 = mean_data1500.rolling(window=window_size).mean()
    min_data1500 = min_data1500.rolling(window=window_size).mean()
    max_data1500 = max_data1500.rolling(window=window_size).mean()

    mean_data2 = data['algorithm.initial_obs_steps: 2 - ' + indicator]
    min_data2 = data['algorithm.initial_obs_steps: 2 - ' + indicator + '__MIN']
    max_data2 = data['algorithm.initial_obs_steps: 2 - ' + indicator + '__MAX']
    mean_data2 = mean_data2.rolling(window=window_size).mean()
    min_data2 = min_data2.rolling(window=window_size).mean()
    max_data2 = max_data2.rolling(window=window_size).mean()

    # Plot the data from both datasets as line plots with shaded regions
    sns.set(font_scale=1.5)
    custom_palette = ['blue', 'orange', 'green']  # 自定义的颜色调色板，可以根据需要添加更多颜色
    sns.set_palette("deep")

    plt.figure(figsize=(6, 5))

    plt.plot(x_data, mean_data15000, label='15000 obs', linewidth=3, marker='*')
    plt.fill_between(x_data,  min_data15000, max_data15000, alpha=0.13)

    plt.plot(x_data, mean_data1500, label='1500 obs', linewidth=3, marker='o')
    plt.fill_between(x_data,  min_data1500, max_data1500, alpha=0.13)

    plt.plot(x_data, mean_data2, label='0 obs', linewidth=3, marker='s')
    plt.fill_between(x_data,  min_data2, max_data2, alpha=0.13)

    if "sis" in file_path:
        plt.ylim(0, 0.0002)
    if "boids" in file_path:
        if "val" in file_path:
            plt.ylim(0, 0.003)
        if "test" in file_path:
            plt.ylim(0.002, 0.006)

    # 使用科学记数法显示 y 轴刻度标签
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    # ax.yaxis.offsetText.set_fontsize(12)  # 调整偏移文本大小
    ax.yaxis.get_major_formatter().set_powerlimits((0, 0))  # 控制指数部分的显示范围

    plt.legend(loc='upper right')
    plt.xlabel('Iteration')
    if "val_score" in indicator:
        plt.ylabel('Validation loss')
    else:
        plt.ylabel('Test loss')
    plt.title('')
    plt.tight_layout()

    base_name = file_path.split('/')[-1]  # 提取最后一个 '/' 之后的部分，得到 'kuramoto_episode_reward_init_obs.csv'
    file_name_without_extension = base_name.split('.')[0]  # 提取 '.' 之前的部分
    plt.savefig("results/" + file_name_without_extension + '.pdf', format='pdf')
    plt.show()

draw_episode_reward(file_path='data/kuramoto_episode_reward_init_obs.csv')
draw_loss(file_path='data/kuramoto_val_loss_init_obs.csv', indicator="model_val_score")
draw_loss(file_path='data/kuramoto_test_loss_init_obs.csv', indicator="model_test_score")

draw_episode_reward(file_path='data/sis_episode_reward_init_obs.csv')
draw_loss(file_path='data/sis_val_loss_init_obs.csv', indicator="model_val_score")
draw_loss(file_path='data/sis_test_loss_init_obs.csv', indicator="model_test_score")

draw_episode_reward(file_path='data/boids_episode_reward_init_obs.csv', nrows=10)
draw_loss(file_path='data/boids_val_loss_init_obs.csv', indicator="model_val_score", nrows=10)
draw_loss(file_path='data/boids_test_loss_init_obs.csv', indicator="model_test_score", nrows=10)