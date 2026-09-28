import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter

def draw_episode_reward(file_path1='data/boids_episode_reward_init_obs.csv',
                        file_path2='data/boids_episode_reward_fixed_node_num.csv',
                        file_path3='data/boids_episode_reward_mlp.csv',
                        file_path4='data/boids_episode_reward_ours_without_ha_gnn.csv',
                        nrows=40
                        ):
    file_path1 = file_path1
    data1 = pd.read_csv(file_path1,nrows=nrows)
    x_data1 = data1['env_step']

    file_path2 = file_path2
    data2 = pd.read_csv(file_path2, nrows=nrows)
    x_data2 = data2['env_step']

    file_path3 = file_path3
    data3 = pd.read_csv(file_path3, nrows=nrows)
    x_data3 = data3['env_step']

    file_path4 = file_path4
    data4 = pd.read_csv(file_path4, nrows=nrows)
    x_data4 = data4['env_step']

    window_size = 4

    mean_data_no_fixed = data1['algorithm.initial_obs_steps: 15000 - episode_reward']
    min_data_no_fixed = data1['algorithm.initial_obs_steps: 15000 - episode_reward__MIN']
    max_data_no_fixed = data1['algorithm.initial_obs_steps: 15000 - episode_reward__MAX']
    mean_data_no_fixed = mean_data_no_fixed.rolling(window=window_size).mean()
    min_data_no_fixed = min_data_no_fixed.rolling(window=window_size).mean()
    max_data_no_fixed = max_data_no_fixed.rolling(window=window_size).mean()

    mean_data_fixed = data2['action_optimizer.is_fixed_control_node_num: true - episode_reward']
    min_data_fixed = data2['action_optimizer.is_fixed_control_node_num: true - episode_reward__MIN']
    max_data_fixed = data2['action_optimizer.is_fixed_control_node_num: true - episode_reward__MAX']
    mean_data_fixed = mean_data_fixed.rolling(window=window_size).mean()
    min_data_fixed = min_data_fixed.rolling(window=window_size).mean()
    max_data_fixed = max_data_fixed.rolling(window=window_size).mean()

    mean_data_mlp = data3['dynamics_model.model_type: mlp - episode_reward']
    min_data_mlp = data3['dynamics_model.model_type: mlp - episode_reward__MIN']
    max_data_mlp = data3['dynamics_model.model_type: mlp - episode_reward__MAX']
    mean_data_mlp = mean_data_mlp.rolling(window=window_size).mean()
    min_data_mlp = min_data_mlp.rolling(window=window_size).mean()
    max_data_mlp = max_data_mlp.rolling(window=window_size).mean()

    mean_data_ours_without_ha_gnn = data4['dynamics_model.model_type: mlp - episode_reward']
    min_data_ours_without_ha_gnn = data4['dynamics_model.model_type: mlp - episode_reward__MIN']
    max_data_ours_without_ha_gnn = data4['dynamics_model.model_type: mlp - episode_reward__MAX']
    mean_data_ours_without_ha_gnn = mean_data_ours_without_ha_gnn.rolling(window=window_size).mean()
    min_data_ours_without_ha_gnn = min_data_ours_without_ha_gnn.rolling(window=window_size).mean()
    max_data_ours_without_ha_gnn = max_data_ours_without_ha_gnn.rolling(window=window_size).mean()

    # Plot the data from both datasets as line plots with shaded regions
    sns.set(font_scale=1.3)
    custom_palette = ['blue', 'orange', 'green']  # 自定义的颜色调色板，可以根据需要添加更多颜色
    sns.set_palette("deep")

    plt.figure(figsize=(6, 5))

    plt.plot(x_data1, mean_data_no_fixed, label='MPCSC', linewidth=3, marker='*')
    plt.fill_between(x_data1,  min_data_no_fixed, max_data_no_fixed, alpha=0.13)

    plt.plot(x_data3, mean_data_mlp, label='MPCSC w/o GNN', linewidth=3, marker='s')
    plt.fill_between(x_data3, min_data_mlp, max_data_mlp, alpha=0.13)

    plt.plot(x_data2, mean_data_fixed, label='MPCSC w/o HA', linewidth=3, marker='o')
    plt.fill_between(x_data2,  min_data_fixed, max_data_fixed, alpha=0.13)

    plt.plot(x_data4, mean_data_ours_without_ha_gnn, label='MPCSC w/o GNN & HA', linewidth=3, marker='^')
    plt.fill_between(x_data4, min_data_ours_without_ha_gnn, max_data_ours_without_ha_gnn, alpha=0.13)

    plt.ylim(50, 140)
    # 使用科学记数法显示 y 轴刻度标签
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    # ax.yaxis.offsetText.set_fontsize(12)  # 调整偏移文本大小
    ax.yaxis.get_major_formatter().set_powerlimits((0, 0))  # 控制指数部分的显示范围

    plt.legend(loc='lower right')
    plt.xlabel('Steps')
    plt.ylabel('Episode Return')
    plt.title('')
    plt.tight_layout()

    base_name = file_path1.split('/')[-1]  # 提取最后一个 '/' 之后的部分，得到 'kuramoto_episode_reward_init_obs.csv'
    file_name_without_extension = base_name.split('_')[0]  # 提取 '.' 之前的部分
    plt.savefig("results/" + file_name_without_extension + '_ablation_episode_reward.pdf', format='pdf')
    plt.show()

def draw_total_cost_reward(file_path1='data/boids_total_cost_reward_init_obs.csv',
                           file_path2='data/boids_total_cost_reward_fixed_node_num.csv',
                           file_path3='data/boids_total_cost_reward_mlp.csv',
                           file_path4='data/boids_total_cost_reward_ours_without_ha_gnn.csv',
                           nrows=40
                           ):
    file_path1 = file_path1
    data1 = pd.read_csv(file_path1,nrows=nrows)
    x_data1 = data1['env_step']

    file_path2 = file_path2
    data2 = pd.read_csv(file_path2, nrows=nrows)
    x_data2 = data2['env_step']

    file_path3 = file_path3
    data3 = pd.read_csv(file_path3, nrows=nrows)
    x_data3 = data3['env_step']

    file_path4 = file_path4
    data4 = pd.read_csv(file_path4, nrows=nrows)
    x_data4 = data4['env_step']

    window_size = 4

    mean_data_no_fixed = data1['algorithm.initial_obs_steps: 15000 - total_cost_reward']
    min_data_no_fixed = data1['algorithm.initial_obs_steps: 15000 - total_cost_reward__MIN']
    max_data_no_fixed = data1['algorithm.initial_obs_steps: 15000 - total_cost_reward__MAX']
    mean_data_no_fixed = mean_data_no_fixed.rolling(window=window_size).mean()
    min_data_no_fixed = min_data_no_fixed.rolling(window=window_size).mean()
    max_data_no_fixed = max_data_no_fixed.rolling(window=window_size).mean()

    mean_data_fixed = data2['action_optimizer.is_fixed_control_node_num: true - total_cost_reward']
    min_data_fixed = data2['action_optimizer.is_fixed_control_node_num: true - total_cost_reward__MIN']
    max_data_fixed = data2['action_optimizer.is_fixed_control_node_num: true - total_cost_reward__MAX']
    mean_data_fixed = mean_data_fixed.rolling(window=window_size).mean()
    min_data_fixed = min_data_fixed.rolling(window=window_size).mean()
    max_data_fixed = max_data_fixed.rolling(window=window_size).mean()

    mean_data_mlp = data3['dynamics_model.model_type: mlp - total_cost_reward']
    min_data_mlp = data3['dynamics_model.model_type: mlp - total_cost_reward__MIN']
    max_data_mlp = data3['dynamics_model.model_type: mlp - total_cost_reward__MAX']
    mean_data_mlp = mean_data_mlp.rolling(window=window_size).mean()
    min_data_mlp = min_data_mlp.rolling(window=window_size).mean()
    max_data_mlp = max_data_mlp.rolling(window=window_size).mean()

    mean_data_ours_without_ha_gnn = data4['dynamics_model.model_type: mlp - total_cost_reward']
    min_data_ours_without_ha_gnn = data4['dynamics_model.model_type: mlp - total_cost_reward__MIN']
    max_data_ours_without_ha_gnn = data4['dynamics_model.model_type: mlp - total_cost_reward__MAX']
    mean_data_ours_without_ha_gnn = mean_data_ours_without_ha_gnn.rolling(window=window_size).mean()
    min_data_ours_without_ha_gnn = min_data_ours_without_ha_gnn.rolling(window=window_size).mean()
    max_data_ours_without_ha_gnn = max_data_ours_without_ha_gnn.rolling(window=window_size).mean()


    # Plot the data from both datasets as line plots with shaded regions
    sns.set(font_scale=1.3)
    custom_palette = ['blue', 'orange', 'green']  # 自定义的颜色调色板，可以根据需要添加更多颜色
    sns.set_palette("deep")

    plt.figure(figsize=(6, 5))

    plt.plot(x_data1, mean_data_no_fixed, label='MPCSC', linewidth=3, marker='*')
    plt.fill_between(x_data1,  min_data_no_fixed, max_data_no_fixed, alpha=0.13)

    plt.plot(x_data3, mean_data_mlp, label='MPCSC w/o GNN', linewidth=3, marker='s')
    plt.fill_between(x_data3, min_data_mlp, max_data_mlp, alpha=0.13)

    plt.plot(x_data2, mean_data_fixed, label='MPCSC w/o HA', linewidth=3, marker='o')
    plt.fill_between(x_data2,  min_data_fixed, max_data_fixed, alpha=0.13)

    plt.plot(x_data4, mean_data_ours_without_ha_gnn, label='MPCSC w/o GNN & HA', linewidth=3, marker='^')
    plt.fill_between(x_data4, min_data_ours_without_ha_gnn, max_data_ours_without_ha_gnn, alpha=0.13)

    plt.ylim(20, 120)
    # 使用科学记数法显示 y 轴刻度标签
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    # ax.yaxis.offsetText.set_fontsize(12)  # 调整偏移文本大小
    ax.yaxis.get_major_formatter().set_powerlimits((0, 0))  # 控制指数部分的显示范围

    plt.legend(loc='upper right')
    plt.xlabel('Steps')
    plt.ylabel('Action Cost')
    plt.title('')
    plt.tight_layout()

    base_name = file_path1.split('/')[-1]  # 提取最后一个 '/' 之后的部分，得到 'kuramoto_episode_reward_init_obs.csv'
    file_name_without_extension = base_name.split('_')[0]  # 提取 '.' 之前的部分
    plt.savefig("results/" + file_name_without_extension + '_ablation_total_cost_reward.pdf', format='pdf')
    plt.show()

def draw_loss(file_path1='data/boids_test_loss_init_obs.csv',
              file_path2='data/boids_test_loss_fixed_node_num.csv',
              file_path3='data/boids_test_loss_mlp.csv',
              file_path4='data/boids_test_loss_ours_without_ha_gnn.csv',
              indicator='model_test_score',
              nrows=40):
    file_path1 = file_path1
    data1 = pd.read_csv(file_path1, nrows=nrows)
    x_data1 = data1['iteration']

    file_path2 = file_path2
    data2 = pd.read_csv(file_path2, nrows=nrows)
    x_data2 = data2['iteration']

    file_path3 = file_path3
    data3 = pd.read_csv(file_path3, nrows=nrows)
    x_data3 = data3['iteration']

    file_path4 = file_path4
    data4 = pd.read_csv(file_path4, nrows=nrows)
    x_data4 = data4['iteration']

    window_size = 4

    mean_data_no_fixed= data1['algorithm.initial_obs_steps: 15000 - ' + indicator]
    min_data_no_fixed = data1['algorithm.initial_obs_steps: 15000 - ' + indicator + '__MIN']
    max_data_no_fixed = data1['algorithm.initial_obs_steps: 15000 - ' + indicator + '__MAX']
    mean_data_no_fixed = mean_data_no_fixed.rolling(window=window_size).mean()
    min_data_no_fixed = min_data_no_fixed.rolling(window=window_size).mean()
    max_data_no_fixed = max_data_no_fixed.rolling(window=window_size).mean()

    mean_data_fixed= data2['action_optimizer.is_fixed_control_node_num: true - ' + indicator]
    min_data_fixed = data2['action_optimizer.is_fixed_control_node_num: true - ' + indicator + '__MIN']
    max_data_fixed = data2['action_optimizer.is_fixed_control_node_num: true - ' + indicator + '__MAX']
    mean_data_fixed = mean_data_fixed.rolling(window=window_size).mean()
    min_data_fixed = min_data_fixed.rolling(window=window_size).mean()
    max_data_fixed = max_data_fixed.rolling(window=window_size).mean()

    mean_data_mlp= data3['dynamics_model.model_type: mlp - ' + indicator]
    min_data_mlp = data3['dynamics_model.model_type: mlp - ' + indicator + '__MIN']
    max_data_mlp = data3['dynamics_model.model_type: mlp - ' + indicator + '__MAX']
    mean_data_mlp = mean_data_mlp.rolling(window=window_size).mean()
    min_data_mlp = min_data_mlp.rolling(window=window_size).mean()
    max_data_mlp = max_data_mlp.rolling(window=window_size).mean()

    mean_data_ours_without_ha_gnn = data4['dynamics_model.model_type: mlp - ' + indicator]
    min_data_ours_without_ha_gnn = data4['dynamics_model.model_type: mlp - ' + indicator + '__MIN']
    max_data_ours_without_ha_gnn = data4['dynamics_model.model_type: mlp - ' + indicator + '__MAX']
    mean_data_ours_without_ha_gnn = mean_data_ours_without_ha_gnn.rolling(window=window_size).mean()
    min_data_ours_without_ha_gnn = min_data_ours_without_ha_gnn.rolling(window=window_size).mean()
    max_data_ours_without_ha_gnn = max_data_ours_without_ha_gnn.rolling(window=window_size).mean()

    # Plot the data from both datasets as line plots with shaded regions
    sns.set(font_scale=1.3)
    custom_palette = ['blue', 'orange', 'green']  # 自定义的颜色调色板，可以根据需要添加更多颜色
    sns.set_palette("deep")

    plt.figure(figsize=(6, 5))

    plt.plot(x_data1, mean_data_no_fixed, label='MPCSC', linewidth=3, marker='*')
    plt.fill_between(x_data1,  min_data_no_fixed, max_data_no_fixed, alpha=0.13)

    plt.plot(x_data3, mean_data_mlp, label='MPCSC w/o GNN', linewidth=3, marker='s')
    plt.fill_between(x_data3, min_data_mlp, max_data_mlp, alpha=0.13)

    plt.plot(x_data2, mean_data_fixed, label='MPCSC w/o HA', linewidth=3, marker='o')
    plt.fill_between(x_data2,  min_data_fixed, max_data_fixed, alpha=0.13)

    plt.plot(x_data4, mean_data_ours_without_ha_gnn, label='MPCSC w/o GNN & HA', linewidth=3, marker='^')
    plt.fill_between(x_data4, min_data_ours_without_ha_gnn, max_data_ours_without_ha_gnn, alpha=0.13)


    # 使用科学记数法显示 y 轴刻度标签
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    # ax.yaxis.offsetText.set_fontsize(12)  # 调整偏移文本大小
    ax.yaxis.get_major_formatter().set_powerlimits((0, 0))  # 控制指数部分的显示范围

    plt.legend(loc='upper right')
    plt.xlabel('Iteration')
    plt.ylabel('Test loss')
    plt.title('')
    plt.tight_layout()

    base_name = file_path1.split('/')[-1]  # 提取最后一个 '/' 之后的部分，得到 'kuramoto_episode_reward_init_obs.csv'
    file_name_without_extension = base_name.split('_')[0]  # 提取 '.' 之前的部分
    plt.savefig("results/" + file_name_without_extension + '_ablation_test_loss.pdf', format='pdf')
    plt.show()

draw_episode_reward(file_path1='data/boids_episode_reward_init_obs.csv',
                    file_path2='data/boids_episode_reward_fixed_node_num.csv',
                    file_path3='data/boids_episode_reward_mlp.csv',
                    file_path4='data/boids_episode_reward_ours_without_ha_gnn.csv',
                    nrows=20)
draw_total_cost_reward(file_path1='data/boids_total_cost_reward_init_obs.csv',
                       file_path2='data/boids_total_cost_reward_fixed_node_num.csv',
                       file_path3='data/boids_total_cost_reward_mlp.csv',
                       file_path4='data/boids_total_cost_reward_ours_without_ha_gnn.csv',
                       nrows=20)
draw_loss(file_path1='data/boids_test_loss_init_obs.csv',
          file_path2='data/boids_test_loss_fixed_node_num.csv',
          file_path3='data/boids_test_loss_mlp.csv',
          file_path4='data/boids_test_loss_ours_without_ha_gnn.csv',
          indicator='model_test_score',
          nrows=20)