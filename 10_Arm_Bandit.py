import numpy as np
import matplotlib.pyplot as plt

import matplotlib.ticker as mtick

num_episode = 2000
plays = 1000


def step(reward_dict, arm):
    return reward_dict[arm][np.random.choice(10)]


def epsilon_greedy_policy(e):
    return 0 if abs(np.random.randn()) <= e else 1


def get_best_machine(best_machine, rewards_dict):
    for k in rewards_dict:
        if best_machine[1] < rewards_dict[k].mean():
            best_machine = (k, rewards_dict[k].mean())
    return best_machine


def get_rewards_dict(action_values):
    rewards_dict = {}
    for a in range(10):
        rewards = np.random.normal(action_values[a], 1, 10)
        rewards_dict[a] = rewards
    return rewards_dict


def n_arm_bandit(e):
    total_rewards = []
    optimal_choice_list = []
    count = 1
    for i in range(num_episode):
        action_values = np.random.normal(0, 1, (10, 1))
        best_machine = (0, 0)

        rewards_dict = get_rewards_dict(action_values)
        best_machine = get_best_machine(best_machine, rewards_dict)

        rewards_per_play = []
        optimal_choice_list_epi = []
        q_values = np.zeros((10, 1))
        action_count = [0] * 10

        for j in range(plays):
            policy = epsilon_greedy_policy(e)
            if policy == 0:
                action = np.random.choice(10)
            else:
                action = np.argmax(q_values)
            action_count[action] += 1
            reward = step(rewards_dict, action)
            if action == best_machine[0]:
                optimal_choice_list_epi.append(1)
            else:
                optimal_choice_list_epi.append(0)
            q_values[action] += (1 / action_count[action]) * (reward - q_values[action])
            count += 1
            rewards_per_play.append(reward)

        total_rewards.append(rewards_per_play)
        optimal_choice_list.append(optimal_choice_list_epi)

    return total_rewards, optimal_choice_list


def get_average_rewards(data):
    np_array = np.array(data)
    return np_array.mean(axis=0)


def get_percentage_optimal_list(data):
    np_array = np.array(data)
    sum_list = np_array.sum(axis=0)
    return sum_list * (100 / num_episode)


rewards_2d, optimal_machine_choices = n_arm_bandit(0)

reward_ep0 = get_average_rewards(rewards_2d)

optimal_choices_ep0 = get_percentage_optimal_list(optimal_machine_choices)

rewards_2d, optimal_machine_choices = n_arm_bandit(0.1)

reward_ep_1 = get_average_rewards(rewards_2d)
optimal_choices_ep1 = get_percentage_optimal_list(optimal_machine_choices)

rewards_2d, optimal_machine_choices = n_arm_bandit(0.01)

reward_ep_2 = get_average_rewards(rewards_2d)
optimal_choices_ep2 = get_percentage_optimal_list(optimal_machine_choices)


def plot_subplot(subplot, x_range, y_data1, y_data2, y_data3, y_label):
    subplot.plot(range(0, x_range), y_data1, c='g', label='eps = 0')
    subplot.plot(range(0, x_range), y_data2, c='k', label='eps = 0.1')
    subplot.plot(range(0, x_range), y_data3, c='r', label='eps = 0.01')
    subplot.set_ylabel(y_label)
    subplot.legend()
    return subplot


fig, (ax1, ax2) = plt.subplots(2)

ax1 = plot_subplot(ax1, plays, reward_ep0, reward_ep_1, reward_ep_2, 'Average rewards over 2000 runs')

ax2.yaxis.set_major_formatter(mtick.PercentFormatter(100))

ax2 = plot_subplot(ax2, plays, optimal_choices_ep0, optimal_choices_ep1, optimal_choices_ep2, 'Optimal Action %')

ax2.set_xlabel('Steps')
fig.show()
