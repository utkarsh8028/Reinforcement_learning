import random

import gym_bandits
import gym
import numpy as np
import pandas as pd

env = gym.make('MultiarmedBandits-v0')




print("action Space: ", env.action_space)
print("ob Space: ", env.observation_space)
Value = np.zeros((10,1))

def epsilon_greedy_policy():
 return  np.argmax(Value)


#for a in range(env.action_space.n):
#    Value = 0.0


alpha = 0.85
gamma = 0.90
num_episode = 100
num_time_steps = 100
s = 0
greedy_ep = 0.05
total_reward=0

def policy_finder(it_num,ep,no_time_steps):
    ep_value = ep*no_time_steps
    #print(Value)
    #print("max",np.argmax(Value))
    #print("random", int(random.uniform(0,len(Value))))
    if it_num <= ep_value:
        print("random")
        m_action = int(random.uniform(0,len(Value)))
    else:
        print("max")
        m_action = int(np.argmax(Value))
    return m_action


for i in range(num_episode):
    env.reset()
    # print(reset)
    for t in range(num_time_steps):
        env.render()
        action1 = epsilon_greedy_policy()
        action = policy_finder(t, greedy_ep, num_time_steps)
        print("action",action," action1", action1)
        s1, reward, done, _ = env.step(action)
        Value[action] += alpha * (reward - Value[action])
        print("action: ", action, "reward: ", reward)
        total_reward += reward
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            print("Done: ", done, reward)
            break
#df = pd.DataFrame(list(Value.items()),columns=['action' , 'value'])
#print(df)
env.close()
print("total reward", total_reward)