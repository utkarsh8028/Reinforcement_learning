import numpy as np
import matplotlib.pyplot as plt


def step(action_values, arm):
    return np.random.normal(action_values[arm], 1, (1, 1))[0]


def epsilon_greedy_policy(action_values, e):
    return np.random.choice(10) if np.random.randn(1) < e else np.argmax(action_values)


num_episode = 1000


def n_arm_bandit(e):
    avg_rewards = []
    action_values = np.random.normal(0, 1, (10, 1))
    g_rewards = 0
    for i in range(1, num_episode + 1):
        total_rewards = 0
        for j in range(1000):
            action = epsilon_greedy_policy(action_values, e)
            reward = step(action_values, action)
            action_values[action] += (1 / i) * (reward - action_values[action])
            total_rewards += reward
        g_rewards += total_rewards
        avg_rewards.append(total_rewards)
        # print("action: ", action, "reward: ", reward)

    print(g_rewards)
    return g_rewards, avg_rewards


reward_1, avg_reward_1 = n_arm_bandit(0)

reward_2, avg_reward_2 = n_arm_bandit(0.01)

reward_3, avg_reward_3 = n_arm_bandit(0.1)

plt.plot(range(0, num_episode), avg_reward_1, c='r')
plt.plot(range(0, num_episode), avg_reward_2, c='g')
plt.plot(range(0, num_episode), avg_reward_3, c='b')

plt.show()
