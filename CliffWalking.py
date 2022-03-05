import gym
import numpy as np

num_episode = 2000
plays = 1000
grid_rows = 4
grid_cols = 12
S = (3, 0)
F = (3, 11)
currentPosition = S
grid = np.zeros((4,12))
grid[3,1:11] = -100
grid[0:-1,]= -1
print('grid',currentPosition[1])

def actionTaken(action):
    allActions = {"up":(currentPosition[0]-1,currentPosition[1]), "down":(currentPosition[0]+1,currentPosition[1]),
                  "left":(currentPosition[0],currentPosition[1]-1),"right":(currentPosition[0]+1,currentPosition[1]+1)}


def cliff_walking_q_learning():
    return

