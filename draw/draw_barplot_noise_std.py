import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
from scipy.stats import ttest_ind

file_path = "data/sis_different_noise.csv"
data = pd.read_csv(file_path, nrows=10)
x_data = data['env_step']


row_nine = data.loc[1]

obs2_data = []
for i in range(1, 13):
    obs2_data.append(row_nine.filter(like="sis_noise_std_obs2_"+str(i)+"-").filter(like="episode_reward")[0])

obs15000_data = []
for i in range(1, 13):
    obs15000_data.append(row_nine.filter(like="sis_noise_std_obs15000_"+str(i)+"-").filter(like="episode_reward")[0])



data = [
    {'obs': '0', 'Network Size': 0, 'Result': obs2_data[0]},
    {'obs': '0', 'Network Size': 0, 'Result': obs2_data[1]},
    {'obs': '0', 'Network Size': 0, 'Result': obs2_data[2]},
    {'obs': '0', 'Network Size': 0, 'Result': obs2_data[3]},
    {'obs': '0', 'Network Size': 0.1, 'Result': obs2_data[4]},
    {'obs': '0', 'Network Size': 0.1, 'Result': obs2_data[5]},
    {'obs': '0', 'Network Size': 0.1, 'Result': obs2_data[6]},
    {'obs': '0', 'Network Size': 0.1, 'Result': obs2_data[7]},
    {'obs': '0', 'Network Size': 0.01, 'Result': obs2_data[8]},
    {'obs': '0', 'Network Size': 0.01, 'Result': obs2_data[9]},
    {'obs': '0', 'Network Size': 0.01, 'Result': obs2_data[10]},
    {'obs': '0', 'Network Size': 0.01, 'Result': obs2_data[11]},
    {'obs': '15000', 'Network Size': 0, 'Result': obs15000_data[0]},
    {'obs': '15000', 'Network Size': 0, 'Result': obs15000_data[1]},
    {'obs': '15000', 'Network Size': 0, 'Result': obs15000_data[2]},
    {'obs': '15000', 'Network Size': 0, 'Result': obs15000_data[3]},
    {'obs': '15000', 'Network Size': 0.1, 'Result': obs15000_data[4]},
    {'obs': '15000', 'Network Size': 0.1, 'Result': obs15000_data[5]},
    {'obs': '15000', 'Network Size': 0.1, 'Result': obs15000_data[6]},
    {'obs': '15000', 'Network Size': 0.1, 'Result': obs15000_data[7]},
    {'obs': '15000', 'Network Size': 0.01, 'Result': obs15000_data[8]},
    {'obs': '15000', 'Network Size': 0.01, 'Result': obs15000_data[9]},
    {'obs': '15000', 'Network Size': 0.01, 'Result': obs15000_data[10]},
    {'obs': '15000', 'Network Size': 0.01, 'Result': obs15000_data[11]},
]


df = pd.DataFrame(data)

data1 = df[:4]['Result'].values
data2 = df[12:16]['Result'].values
ttest_result = ttest_ind(data1, data2)
print("ttest_result1 = ", ttest_result)

data1 = df[4:8]['Result'].values
data2 = df[16:20]['Result'].values
ttest_result = ttest_ind(data1, data2)
print("ttest_result2 = ", ttest_result)

data1 = df[8:12]['Result'].values
data2 = df[20:24]['Result'].values
ttest_result = ttest_ind(data1, data2)
print("ttest_result3 = ", ttest_result)

sns.set(style="white", font_scale=1.5)
plt.figure(figsize=(8, 6))
custom_palette = ["#1f77b4", "#ff7f0e"]
sns.barplot(x="Network Size", y="Result", hue="obs", data=df) #, palette=custom_palette
plt.xlabel("Noise")
plt.ylabel("Episode Return")
plt.legend(title="obs", loc="upper right")

plt.savefig("results/sis_noise.pdf", format='pdf')
plt.show()