import gym_bandits
import gym
import pandas as pd

env = gym.make('MultiarmedBandits-v0')




print("action Space: ", env.action_space)
print("ob Space: ", env.observation_space)
Value = {}

def epsilon_greedy_policy():
    return  max(Value, key= lambda x: Value[x])


for a in range(env.action_space.n):
    Value[a] = 0.0


alpha = 0.85
gamma = 0.90
num_episode = 1000
num_time_steps = 100
s = 0

for i in range(num_episode):

    env.reset()
    # print(reset)
    for t in range(num_time_steps):
        env.render()
        action = epsilon_greedy_policy()
        s1, reward, done, _ = env.step(action)
        Value[action] += alpha * (reward - Value[action])
        print("action: ", action, "reward: ", reward)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            print("Done: ", done, reward)
            break
df = pd.DataFrame(list(Value.items()),columns=['state' , 'value'])
print(df)
env.close()