import random

import numpy as np
import matplotlib.pyplot as plt


def step(action_values, arm):
    return np.random.normal(action_values[arm], 1, (1, 1))[0]


def epsilon_greedy_policy(action_values, e):
    return np.random.choice(10) if np.random.randn(1) < e else np.argmax(action_values)


def policy_finder( ep_value, rand_limit, action_values):
    if rand_limit <= ep_value:
        m_action = [int(random.uniform(0, len(action_values))), int(np.argmax(action_values))]
        res = int(random.uniform(0, 1))
        if res == 0:
            rand_limit += 1
        policy = m_action[res]
    else:
        print("max")
        policy = int(np.argmax(action_values))
    return policy


num_episode = 1000


def n_arm_bandit(e):
    avg_rewards = []

    g_rewards = 0
    rand_limit = 0
    num_plays = 2000
    for i in range(1, num_episode + 1):
        action_values = np.random.normal(0, 1, (10, 1))
        q_value = np.zeros((10,1))
        total_rewards = 0
        for j in range(0, num_plays):
            action = policy_finder(e*num_plays, rand_limit, action_values)
            reward = step(action_values, action)
            q_value[action] += (1 / i) * (reward - q_value[action])
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
