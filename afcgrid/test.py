import torch
import numpy as np
from environment import AfcEnvGrid
from model import ActorCriticModel

model_path = "./saved_model/model_seed42"
model = ActorCriticModel()
model.load_state_dict(torch.load(model_path))

env = AfcEnvGrid()

def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)
    m = torch.distributions.Categorical(probs)
    action = m.sample()
    return action.item(), probs.to('cpu').detach().numpy().copy()


env.reset()
state = env.reset()
for _ in range(200):
    env.render(mode="rgb_array")

    action, probs = select_action(state)
    # action = env.action_space.sample()  # Actions sampled at random

    state, reward, done, _ = env.step(action)
    
    print(action, env.fixs, reward, probs)

    if done:
        env.reset()
