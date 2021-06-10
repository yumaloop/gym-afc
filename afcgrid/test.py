import torch
from model import ActorCriticModel

model_path = "./saved_model/model_seed42"
model = ActorCriticModel()
model.load_state_dict(torch.load(model_path))


