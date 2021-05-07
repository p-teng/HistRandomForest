import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import openml

metadata = ["HistRF_metadata", "BagDT_metadata", "RF_metadata"]
def unpickle(path):
    infile = open(path, 'rb')
    new_dict = pickle.load(infile)
    infile.close()
    return new_dict 

def get_train_time(clf, data):
    for name in metadata:
        if name == clf:
            return np.average(data[name]['train_times'])
    
def get_tasks(results_dir):
    tasks = []
    # task_score_means = []
    # task_score_stds = []
    for file in os.listdir(results_dir)[::-1]:
        with open(results_dir + file, 'rb') as f:
            results_dict = pickle.load(f)
        # for k in ['task_id', 'task', 'n_samples', 'n_classes', 'n_features']:
        #     tasks.append(k)
        tasks.append(
            {k: results_dict[k] for k in ['task_id', 'task', 'n_samples', 'n_classes', 'n_features']
        })
        tasks[-1]['BagDT_time'] = get_train_time(metadata[0], results_dict)
        tasks[-1]['RF_time'] = get_train_time(metadata[1], results_dict)
        tasks[-1]['HRF_time'] = get_train_time(metadata[2], results_dict)
    return tasks


# header = ['Dataset', 'n_classes', 'n_samples', 'n_features'] + clfs

# score_df = pd.DataFrame(
#         df_rows, columns=header, index=[task['task_id'] for task in tasks]
#     ).sort_index()

# dt = unpickle("./cc18_results/results_cv/adult_results_dict.pkl")
# f, ax = plt.subplots(figsize=(12, 16))
# sns.despine(bottom=True, left=True)
# HistRF = get_train_time(metadata[0], dt)
# BagDT = get_train_time(metadata[1], dt)
# RF = get_train_time(metadata[2], dt)
# df = pd.DataFrame([HistRF, BagDT, RF])
# sns.stripplot(
#               data = df, dodge=False, alpha=1, zorder=1, palette=color_dict)
# plt.savefig(f'./figures/time/test.pdf')
# plt.show()
results_dir = f'./cc18_results/results_cv/'
task_list = get_tasks(results_dir)
# print("TASKS: " + str(task_list))
# for file in os.listdir(results_dir)[::-1]:
#     with open(results_dir + file, 'rb') as f:
#         results_dict = pickle.load(f)
#         df_arr[k] = [results_dir['task'], "BagDT", get_train_time(metadata[0], results_dict)]
#         hrf_train_times.append(get_train_time(metadata[0], results_dict))
#         bagdt_train_times.append(get_train_time(metadata[1], results_dict))
#         rf_train_times.append(get_train_time(metadata[2], results_dict))
#     k += 1
#     ids.append(k)

df = pd.DataFrame(task_list)
print(df.head)
f, ax = plt.subplots(figsize=(12, 16))
ax.set_xscale('log')
data = ["HRF", "BagDT", "RF"]
sns.stripplot(x="HRF_time", y="task",
            data=df, dodge=False, color="green", alpha=1, zorder=1)
sns.stripplot(x="BagDT_time", y="task",
            data=df, dodge=False, color="red", alpha=1, zorder=1)
sns.stripplot(x="RF_time", y="task",
            data=df, dodge=False, color="blue", alpha=1, zorder=1)
ax.set_xlabel("Training time")
plt.savefig(f'./figures/train_time_scatterplot.pdf')
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='blue', edgecolor='b',
                         label='RF'),
                   Patch(facecolor='red', edgecolor='r',
                         label='BagDT'),
                   Patch(facecolor='green', edgecolor='g',
                         label='HistRF')]
ax.legend(handles=legend_elements, loc='center')
plt.show()