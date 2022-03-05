import gym
import numpy as np

num_episode = 2000
plays = 1000
grid_rows = 4
grid_cols = 12
S = (3, 0)
F = (3, 11)
grid = np.zeros((4,12))
grid[3,1:11] = -100
grid[0:-1,]= -1
print('grid',grid)
def cliff_walking_q_learning():
    return

