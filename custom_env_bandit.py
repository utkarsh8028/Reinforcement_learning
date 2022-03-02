import numpy as np
import matplotlib.pyplot as plt


# q_values = np.zeros((10, 1))

def step(action_values, arm):
    return np.random.normal(action_values[arm], 1, 1)


def epsilon_greedy_policy( e):
    return 0 if np.random.randn(1) < e else 1


num_episode = 2000



def n_arm_bandit(e):
    avg_rewards = []
    count = 1
    for i in range(num_episode):
        action_values = np.random.normal(0, 1, (10, 1))
        rewards_per_epi = []
        q_values = np.zeros((10, 1))
        action_count = [0] * 10
        reward_per_machine = np.zeros((10, 1))
        for j in range(1000):
            policy = epsilon_greedy_policy(e)
            action = np.random.choice(10) if policy == 0 else np.argmax(q_values)
            #reward_per_machine[action] += reward_per_machine[action]/action_count[action]
            action_count[action] += 1
            reward = step(action_values, action)
            q_values[action] += (1 / action_count[action]) * (reward - q_values[action])
            count += 1
            rewards_per_epi.append(reward)
        avg_rewards.append(rewards_per_epi)
        # print("action: ", action, "reward: ", reward)

    return avg_rewards


def get_average_rewards(data):
    np_array = np.array(data)
    return np_array.mean(axis=0)


avg_reward_1 = get_average_rewards(n_arm_bandit(0))

avg_reward_2 = get_average_rewards(n_arm_bandit(0.01))

avg_reward_3 = get_average_rewards(n_arm_bandit(0.1))

plt.plot(range(0, 1000), avg_reward_1, c='r')
plt.plot(range(0, 1000), avg_reward_2, c='g')
plt.plot(range(0, 1000), avg_reward_3, c='b')

plt.show()
