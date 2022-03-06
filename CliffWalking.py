import gym
import numpy as np


class Environment:

    grid_rows = 4
    grid_cols = 12
    grid = np.zeros((grid_rows, grid_cols))
    start = (0, 0)
    finish = (0, 12)
    current_position = start
    all_actions = {"up": (current_position[0] - 1, current_position[1]),
                   "down": (current_position[0] + 1, current_position[1]),
                   "left": (current_position[0], current_position[1] - 1),
                   "right": (current_position[0], current_position[1] + 1)}

    def __init__(self, position):
        self.current_position = position

    def environment_returns(self, action):

        reward = -100 if self.current_position[0] == 0 and self.current_position[1] in range(1, 11) else -1

        print('grid', self.grid)
        state = self.action_taken(action)
        return reward, state

    def action_taken(self, action):
        if action == "up":
            position = self.current_position[0] - 1, self.current_position[1]
        elif action == "down":
            position = self.current_position[0] + 1, self.current_position[1]
        elif action == "left":
            position = self.current_position[0], self.current_position[1] - 1
        elif action == "right":
            position = self.current_position[0], self.current_position[1] + 1
        else:
            position = self.current_position[0], self.current_position[1]

        return position


env = Environment((6,0))
rewards, states = env.environment_returns("down")
print(rewards, states)


def sarsa_policy(e):
    return 0 if abs(np.random.randn()) <= e else 1


num_episode = 5
num_play = 10


def cliff_walking_sarsa(e):

    total_reward = []
    print("hi")
    for i in range(num_episode):
        env.reset()
        Q = {}
        print("hi2")
        for j in range(num_play):
            print("hi3")
            policy = sarsa_policy(e)
            reward = env.step(policy)
            print("range", env.reward_range)
            print("reward", reward)
    return 0

