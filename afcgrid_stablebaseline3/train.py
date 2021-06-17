import gym
from stable_baselines3 import DQN
from environment import AfcEnvGrid

env = AfcEnvGrid()
model = DQN("MlpPolicy", env, verbose=2)
model.learn(total_timesteps=300000, log_interval=10)
model.save("dqn-fix2")
