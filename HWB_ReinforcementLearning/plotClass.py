import json

import numpy as np
import matplotlib.pyplot as plt
import csv


plot_data = []
results_to_plot = ["1a", "1b", "1c"]

for i in range(len(results_to_plot)):
    filename = "datafiles/data_" + results_to_plot[i]
    plot_data.append([])
    with open(filename, "r") as f:
        reader = csv.reader(f, delimiter=',')
        plot_data[i] = [[float(value) for value in row] for row in reader]

    # Computing the 100 episode average caveman style:
    data_sum = 0
    for j in range(len(plot_data[i])):
        data_sum += plot_data[i][j][1]
        if j >= 100:
            data_sum -= plot_data[i][j-100][1]
        plot_data[i][j].append(data_sum/min(j+1, 100))

for i in range(len(results_to_plot)):
    episodes = [sublist[0] for sublist in plot_data[i]]
    rewards = [sublist[1] for sublist in plot_data[i]]
    reward_sum = [sublist[2] for sublist in plot_data[i]]
    plt.plot(episodes, rewards, label='reward')
    plt.plot(episodes, reward_sum, label='reward averaged ove 100 episodes')
    plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('Total reward')
    plt.title('Reward development task' + results_to_plot[i])
    plt.show()
