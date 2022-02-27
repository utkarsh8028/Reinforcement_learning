import numpy as np
import matplotlib.pyplot as plt

action_values = np.random.normal(0, 1, (10, 1))


class Env(object):
    rewards = []

    def __init__(self, values):

        self.values = values

        for v in self.values:
            self.rewards.append(np.random.normal(v, 1, (1, 1))[0])

    def step(self, arm):
        return self.rewards[arm]


def epsilon_greedy_policy(e):
    return np.random.choice(10) if abs(np.random.randn()) < e else np.argmax(action_values)


alpha = 0.85
gamma = 0.90
num_episode = 20000
s = 0


env = Env(action_values)


def n_arm_bandit(e):
    avg_rewards = []
    total_rewards = 0
    for i in range(1, num_episode + 1):

        action = epsilon_greedy_policy(e)
        reward = env.step(action)
        action_values[action] += (1 / i) * (reward - action_values[action])
        total_rewards += reward
        avg_rewards.append(total_rewards / i)
        # print("action: ", action, "reward: ", reward)

    print(total_rewards)

    return total_rewards, avg_rewards


reward_1, avg_reward_1 = n_arm_bandit(0)

reward_2, avg_reward_2 = n_arm_bandit(0.01)

reward_3, avg_reward_3 = n_arm_bandit(0.1)

plt.plot(range(0, num_episode), avg_reward_1, c='r')
plt.plot(range(0, num_episode), avg_reward_2, c='g')
plt.plot(range(0, num_episode), avg_reward_3, c='b')

plt.show()

