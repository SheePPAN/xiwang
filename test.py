# /usr/bin/env python3

import gymnasium as gym
env = gym.make('Acrobot-v1', render_mode="rgb_array")
#env = gym.make('Acrobot-v1')
#env = gym.make('MountainCar-v0')
#env = gym.make('Pendulum-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, _,info = env.step(action)
        print(observation)
        print(reward, done, info)
        print(action)
        print("Episode finished after {} timesteps".format(t+1))