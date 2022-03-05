import numpy as np
import matplotlib.pyplot as plt


# q_values = np.zeros((10, 1))

def step(action_values, arm):
    return np.random.normal(action_values[arm], 1, 1)


def epsilon_greedy_policy(e):
    return 0 if abs(np.random.randn()) <= e else 1


num_episode = 2000
plays = 10000


def n_arm_bandit(e):
    total_rewards = []
    count = 1
    for i in range(num_episode):
        action_values = np.random.normal(0, 1, (10, 1))
        rewards_per_play = []
        q_values = np.zeros((10, 1))
        action_count = [0] * 10
        reward_per_machine = np.zeros((10, 1))
        for j in range(plays):
            action = 0
            if e != 0:
                policy = epsilon_greedy_policy(e)
                action = np.random.choice(10) if policy == 0 else np.argmax(q_values)
            # reward_per_machine[action] += reward_per_machine[action]/action_count[action]
            else:
                action = np.argmax(q_values)
            action_count[action] += 1
            reward = step(action_values, action)
            q_values[action] += (1 / action_count[action]) * (reward - q_values[action])
            count += 1
            rewards_per_play.append(reward)
        total_rewards.append(rewards_per_play)
        # print("action: ", action, "reward: ", reward)

    return total_rewards


def get_average_rewards(data):
    np_array = np.array(data)
    return np_array.mean(axis=0)


reward_ep0 = get_average_rewards(n_arm_bandit(0))

reward_ep_1 = get_average_rewards(n_arm_bandit(0.01))

reward_ep_2 = get_average_rewards(n_arm_bandit(0.1))

plt.plot(range(0, plays), reward_ep0, c='g')
plt.plot(range(0, plays), reward_ep_1, c='r')
plt.plot(range(0, plays), reward_ep_2, c='k')

plt.show()
