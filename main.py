import time
import gym_bandits
import gym

env = gym.make('CartPole-v0')
counter=0
for i_episode in range(20000):
    observation = env.reset()
    for t in range(1000):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            counter+=1
            print("Episode finished after {} timesteps".format(t+1))
            print(info)
            print(counter)
            break
env.close()