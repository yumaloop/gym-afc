import numpy as np
from environment import AfcEnv

env = AfcEnv()
env.reset(init_pos=[0, 0])
observation = env.reset()
for _ in range(10000):
    env.render(mode="rgb_array")
    action = env.action_space.sample()  # random action
    observation, reward, done, _ = env.step(np.array([1.0, 0.5, 0.7]))
    # observation, reward, done, _ = env.step(env.action_space.sample())

    if done:
        env.reset()
