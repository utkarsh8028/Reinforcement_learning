import random

import numpy as np
import matplotlib.pyplot as plt
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


# remove impossible next positions like -1,0 or 0,-1 etc
def filter_positions(position):
    next_positions = [("up", position[0] - 1, position[1]), ("down", position[0] + 1, position[1]),
                      ("left", position[0], position[1] - 1), ("right", position[0], position[1] + 1)]
    next_positions = list(filter(lambda x: (-1 < x[1] < 4 and -1 < x[2] < 12), next_positions))
    pos = []
    for p in next_positions:
        if p[1] == 4:
            print('2')
        if p[1] == 3 and p[2] == 0:
            pass
        else:
            pos.append(p)

    return pos


# get max action
def max_action(position, q_values):
    next_positions = filter_positions(position)

    action = max(next_positions, key=lambda item: q_values[position][item[0]])
    return action[0]


def random_action(position):
    next_positions = filter_positions(position)

    return random.choice(next_positions)[0]


def update_q_value(q_value, reward, current_position, c_action, next_position, n_action):
    q_value[current_position][c_action] += (
            alpha * (reward + (gamma * q_value[next_position][n_action]) - q_value[current_position][c_action]))
    # print("q", q_value[current_position][c_action])
    return q_value


def init_q_values():
    q_dict = {}
    for i in range(4):
        for j in range(12):
            q_dict[(i, j)] = {}
            q_dict[(i, j)]["up"] = 0
            q_dict[(i, j)]["down"] = 0
            q_dict[(i, j)]["left"] = 0
            q_dict[(i, j)]["right"] = 0
    return q_dict


def cliff_walking_sarsa(e=0.1):
    final_rewards = []
    path = []
    np.random.seed(0)
    for j in range(num_play):
        total_reward = []
        env = Environment()
        q_value = init_q_values()
        path = []
        count = 0
        for i in range(num_episode):
            current_position = env.start
            path = [current_position]
            count = 0
            reward_per_epi = 0
            c_action = random_action(current_position) if abs(np.random.randn()) <= e else max_action(current_position,
                                                                                                      q_value)
            while current_position != env.finish:

                reward, next_state = env.environment_returns(c_action, current_position)
                n_action = random_action(next_state) if sarsa_policy(e) == 0 else max_action(next_state, q_value)
                q_value = update_q_value(q_value, reward, current_position, c_action, next_state, n_action)
                reward_per_epi += reward
                if reward == -100:
                    current_position = env.start
                    c_action = random_action(current_position) if abs(np.random.randn()) <= e else max_action(
                        current_position,
                        q_value)
                    continue
                current_position = next_state
                path.append(current_position)
                c_action = n_action
            if current_position == env.finish:
                count += 1
            # print('reward per episode',reward_per_epi)
            total_reward.append(reward_per_epi)
        final_rewards.append(total_reward)

        # print("count of goal reached", count)
    # print("finished", method, path)
    return final_rewards, path


def cliff_walking_q_learning(e=0.0):
    final_rewards = []
    path = []
    np.random.seed(0)
    for j in range(num_play):
        total_reward = []
        env = Environment()
        q_value = init_q_values()
        count = 0
        for i in range(num_episode):
            # print("episode: ", i)
            current_position = env.start
            path = [current_position]
            reward_per_epi = 0
            while current_position != env.finish:
                c_action = random_action(current_position) if abs(np.random.randn()) < e else max_action(
                    current_position,
                    q_value)
                reward, next_state = env.environment_returns(c_action, current_position)
                n_action = max_action(next_state, q_value)
                q_value = update_q_value(q_value, reward, current_position, c_action, next_state, n_action)
                reward_per_epi += reward
                if reward == -100:
                    current_position = env.start
                    continue
                current_position = next_state
                path.append(current_position)
            if current_position == env.finish:
                count += 1
            # print('reward per episode',reward_per_epi)
            total_reward.append(reward_per_epi)
        final_rewards.append(total_reward)
    return final_rewards, path



def moving_average(x, w):
    arr = np.array(x).mean(axis=0)
    return np.convolve(arr, np.ones(w), 'same') / w


rewards_sarsa, sarsa_path = cliff_walking_sarsa(0.1)

rewards_q, q_path = cliff_walking_q_learning(0.1)

print("sarsa path with e 0.1 ", sarsa_path)
print("q path with e 0.1", q_path)


plt.plot(range(num_episode), moving_average(rewards_sarsa, 10), c='g', label='sarsa')
plt.plot(range(num_episode), moving_average(rewards_q, 10), c='r', label='q-learning')
plt.ylim(-100, 0)
plt.legend()
plt.xlabel('Epsilon 0.1')

plt.show()

rewards_sarsa, sarsa_path = cliff_walking_sarsa( 0.01)

rewards_q, q_path = cliff_walking_q_learning( 0.01)

print("sarsa path with e 0.01", sarsa_path)
print("q path with e 0.01", q_path)


plt.plot(range(num_episode), moving_average(rewards_sarsa, 10), c='g', label='sarsa ')
plt.plot(range(num_episode), moving_average(rewards_q, 10), c='r', label='q-learning ')
plt.ylim(-100, 0)
plt.legend()
plt.xlabel('Epsilon 0.01')
plt.show()

rewards_sarsa, sarsa_path = cliff_walking_sarsa(0)

rewards_q, q_path = cliff_walking_q_learning(0)

print("sarsa path with e 0", sarsa_path)
print("q path with e 0", q_path)


plt.plot(range(num_episode), moving_average(rewards_sarsa, 10), c='g', label='sarsa')
plt.plot(range(num_episode), moving_average(rewards_q, 10), c='r', label='q-learning')
plt.ylim(-100, 0)
plt.legend()
plt.xlabel('Epsilon 0')
plt.show()
