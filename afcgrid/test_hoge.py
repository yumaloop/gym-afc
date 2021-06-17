import gym
from stable_baselines3 import DQN
from environment import AfcEnvGrid

env = AfcEnvGrid()
model = DQN.load("dqn")

obs = env.reset()
for i in range(10):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    
    print(env.fixs, reward)
    
    env.render()
    if done:
      obs = env.reset()
