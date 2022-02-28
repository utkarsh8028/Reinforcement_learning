import random

import gym_bandits
import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

env = gym.make('MultiarmedBandits-v0')

print("action Space: ", env.action_space)
print("ob Space: ", env.observation_space)
Value = np.zeros((10,1))


def epsilon_greedy_policy():
    return  np.argmax(Value)


#for a in range(env.action_space.n):
#    Value = 0.0


gamma = 0.90
num_episode = 100
num_time_steps = 100
s = 0


env.reset()


def policy_finder(it_num, ep_value, rand_limit):
    if rand_limit <= ep_value:
        m_action = [int(random.uniform(0, len(Value))), int(np.argmax(Value))]
        res = int(random.uniform(0, 1))
        if res == 0:
            rand_limit += 1
        policy = m_action[res]
    #if it_num <= ep_value:
    #   policy = int(random.uniform(0, len(Value)))
    else:
        print("max")
        policy = int(np.argmax(Value))
    return policy


def n_arm_bandit_impl(greedy_ep):
    ep_value = greedy_ep*num_time_steps
    rand_limit=0
    total_reward = 0
    avg_rewards = []

    # print(reset)
    for t in range(1,num_time_steps+1):
        alpha = 1 / t
        env.render()
        action1 = epsilon_greedy_policy()

        #action = policy_finder(t, greedy_ep, num_time_steps)
        action = policy_finder(t, ep_value,rand_limit)
        print("action",action," action1", action1)
        s1, reward, done, _ = env.step(action)
        Value[action] += alpha * (reward - Value[action])
        print("action: ", action, "reward: ", reward)
        total_reward += reward
        avg_rewards.append(total_reward / t)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            print("Done: ", done, reward)
            break


    print("total reward", total_reward)
    return avg_rewards,total_reward


avgr1,tr1=n_arm_bandit_impl(0.05)
plt.plot(range(0, num_episode), avgr1, c='b')

avgr2,tr2=n_arm_bandit_impl(0.0)
plt.plot(range(0, num_episode), avgr2, c='g')

avgr3,tr3=n_arm_bandit_impl(0.1)
plt.plot(range(0, num_episode), avgr3, c='r')

avgr4,tr4=n_arm_bandit_impl(1)
plt.plot(range(0, num_episode), avgr4, c='y')
plt.show()
print("total rewards",tr1,tr2,tr3,tr4)
env.close()