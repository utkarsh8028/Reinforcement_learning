import gym_bandits
import gym

env = gym.make('MultiarmedBandits-v0')


def epsilon_greedy_policy(state):
    return env.action_space.sample()


print("action Space: ",env.action_space)
print("ob Space: ",env.observation_space)
Value = {}
for s in range(env.observation_space.n):
    Value[s] = 0.0

alpha=0.85
gamma=0.90
num_episode = 1000
num_time_steps = 100

for i in range(num_episode):

    reset = env.reset()
   # print(reset)
    for t in range(num_time_steps):
        env.render()
        policy = epsilon_greedy_policy()
        s1, reward, done, _ = env.step(policy)
        Value[s] += alpha * (reward + gamma * (Value[s1]-Value[s]))
        print("s1: ", s1, "s: ", s)
        s = s1
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            print("Done: ", done, reward)
            break