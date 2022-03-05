import gym
import numpy as np

num_episode = 5
num_play = 10


def environment(current_position, action):
    grid_rows = 4
    grid_cols = 12
    start = (0, 0)
    finish = (0, 12)
    grid = np.zeros((grid_rows, grid_cols))
    all_actions = {"up": (current_position[0] - 1, current_position[1]),
                  "down": (current_position[0] + 1, current_position[1]),
                  "left": (current_position[0], current_position[1] - 1),
                  "right": (current_position[0], current_position[1] + 1)}

    reward = -100 if current_position[0] == 0 and current_position[1] in range(1, 11) else -1

    print('grid', grid)
    state = action_taken(action, current_position, all_actions)
    return reward, state


def action_taken(action, current_position,all_action):
    position = all_action[action]
    print(position)
    return position


environment((6,0),"right")

def sarsa_policy(e):
    print("no hi")
    return 0 if abs(np.random.randn()) <= e else 1


def cliff_walking_sarsa(e):
    total_reward = []
    env = environment()
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


#cliff_walking_sarsa(0.01)
