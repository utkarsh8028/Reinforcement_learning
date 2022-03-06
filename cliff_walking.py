import random

import numpy as np


num_episode = 500
num_play = 10
alpha = 0.1
gamma = 1


class Environment:
    grid_rows = 4
    grid_cols = 12
    grid = np.zeros((grid_rows, grid_cols))
    start = (3, 0)
    finish = (3, 11)
    current_position = start
    all_actions = ["up", "down", "left", "right"]

    def environment_returns(self, action, current_position):
        self.current_position = current_position
        # print('grid', self.grid)
        state = self.action_taken(action)
        reward = -100 if state[0] == 3 and state[1] in range(1, 11) else -1
        return reward, state

    def action_taken(self, action):
        if action == self.all_actions[0]:
            position = self.current_position[0] - 1, self.current_position[1]
        elif action == self.all_actions[1]:
            position = self.current_position[0] + 1, self.current_position[1]
        elif action == self.all_actions[2]:
            position = self.current_position[0], self.current_position[1] - 1
        elif action == self.all_actions[3]:
            position = self.current_position[0], self.current_position[1] + 1
        else:
            position = self.current_position

        return position


def sarsa_policy(e):
    return 0 if abs(np.random.randn()) <= e else 1


def max_action(position, q_values):

    next_positions = filter_positions(position)
    if len(next_positions) == 0:
        print('s')
    action = max(next_positions, key=lambda item: q_values[item[1], item[2]])
    # action = max(next_positions, key=lambda k: next_positions[k])
    print("ac", action)
    return action[0]


def update_sarsa_q_value(q_value, reward, current_pos, next_pos):
    i = next_pos[0]
    j = next_pos[1]
    x = current_pos[0]
    y = current_pos[1]
    q_value[i][j] = q_value[i][j] + alpha * (reward + (gamma * (q_value[x][y])) - q_value[i][j])
    print("q", q_value[x][y])
    return q_value


def filter_positions(position):
    next_positions = [("up", position[0] - 1, position[1]), ("down", position[0] + 1, position[1]),
                      ("left", position[0], position[1] - 1), ("right", position[0], position[1] + 1)]
    next_positions = list(filter(lambda x: (-1 < x[1] < 4 and -1 < x[2] < 12), next_positions))
    pos =[]
    for p in next_positions:
        if p[1]==0 and p[2] ==0:
           pass
        else:
            pos.append(p)
    return pos


def random_action(position):

    next_positions = filter_positions(position)

    return random.choice(next_positions)[0]


def cliff_walking(method, e=0.0):
    total_reward = []
    env = Environment()
    print(env.grid)
    q_value = np.zeros(env.grid.shape)
    for i in range(num_episode):
        print("episode: ", i)
        current_position = env.start
        count = 0
        reward_per_epi = 0
        while current_position != env.finish:
            policy = 1 if method == "Q-Learning" else sarsa_policy(e)
            action = random_action(current_position) if policy == 0 else max_action(current_position, q_value)
            reward, next_position = env.environment_returns(action, current_position)
            print("policy,reward,next position", policy, reward, next_position)
            q_value = update_sarsa_q_value(q_value, reward, current_position, next_position)
            print("q matrix \n", q_value)
            reward_per_epi += reward
            if reward == -100:
                print("restarting")
                break
            current_position = next_position
            count += 1
        if current_position == env.finish:
            print("yuhuuuu")
        total_reward.append(reward_per_epi)

    print("finished", method)
    return total_reward


rewards_sarsa = cliff_walking("sarsa", 0.1)
rewards_q = cliff_walking("Q-Learning")

import matplotlib.pyplot as plt

plt.plot(range(num_episode),rewards_sarsa, c='g', label='sara = 0')
plt.plot(range(num_episode),rewards_q, c='r', label='eps = 0')

plt.show()
