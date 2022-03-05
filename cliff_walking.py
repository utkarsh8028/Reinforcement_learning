import random

import numpy as np


class Environment:
    grid_rows = 4
    grid_cols = 12
    grid = np.zeros((grid_rows, grid_cols))
    start = (0, 0)
    finish = (0, 12)
    current_position = start
    all_actions = ["up", "down", "left", "right"]

    def environment_returns(self, action, current_position):
        self.current_position = current_position
        reward = -100 if self.current_position[0] == 0 and self.current_position[1] in range(1, 12) else -1
        # print('grid', self.grid)
        state = self.action_taken(action)
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


num_episode = 5
num_play = 10
alpha = 0.1
gamma = 1

def max_action():
    return "up"


def update_sarsa_q_value(q_value, reward, current_pos, next_pos):
    i = next_pos[0]
    j = next_pos[1]
    x = current_pos[0]
    y = current_pos[1]
    q_value[i][j] = q_value[i][j] + alpha*(reward+gamma*(q_value[x][y])-q_value[x][y])
    print("sarsa q", q_value[i][j])
    return q_value


def cliff_walking(e, method):
    total_reward = []
    env = Environment()
    print(env.grid)
    q_value = np.zeros(env.grid.shape)
    for i in range(num_episode):
        current_position = env.start
        for j in range(num_play):
            policy = 1 if method == "Q-Learning" else sarsa_policy(e)
            action = random.choice(env.all_actions) if policy == 0 else max_action()
            reward, next_position = env.environment_returns(action, current_position)
            print("reward", policy, reward, next_position)
            q_value = update_sarsa_q_value(q_value, reward, current_position, next_position)
            print(q_value)
            current_position = next_position
    return 0





def q_value_without_epsilon(q_values,reward,current_position):
    max_q_value=max((q_values[current_position[0]+1,current_position[1]]),
                    (q_values[current_position[0]-1,current_position[1]]),
                    (q_values[current_position[0],current_position[1]+1]),
                    (q_values[current_position[0],current_position[1]-1]))
    q_values[current_position] = q_values[current_position] + alpha[reward+ (gamma * max_q_value) - q_values[current_position]]

q_value_without_epsilon((1,1))
cliff_walking(0.5, "sarsa")
cliff_walking(0.5, "Q-Learning")
