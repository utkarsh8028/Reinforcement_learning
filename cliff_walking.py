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


def cliff_walking(e, method):
    total_reward = []
    env = Environment()
    print(env.grid)
    for i in range(num_episode):
        current_position = env.start
        q_value = {}
        for j in range(num_play):
            policy = 1 if method == "Q-Learning" else sarsa_policy(e)
            action = random.choice(env.all_actions) if policy == 0 else max_action()
            reward, position = env.environment_returns(action, current_position)
            print("reward", policy, reward, position)
            current_position = position
    return 0

def qvaluewithoutepsilon(reward):
    current_position = (1,1)
    q_values = np.zeros((4, 12))
    q_values[current_position] = q_values[current_position] + alpha[reward]
    print(q_values)
cliff_walking(0.5, "sarsa")
cliff_walking(0.5, "Q-Learning")
