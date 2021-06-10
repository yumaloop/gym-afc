import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCriticModel(nn.Module):
    """
    implements both actor and critic in one model
    https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py
    """
    def __init__(self):
        super(Policy, self).__init__()
        action_dims = 8
        state_dims = 2
        h_dims = 8
        
        self.affine1 = nn.Linear(state_dims, 16, bias=False)
        self.affine2 = nn.Linear(16, 16, bias=False)
        
        self.hx = torch.zeros(1, h_dims)
        self.cx = torch.zeros(1, h_dims)
        self.lstmcell = nn.LSTMCell(16, h_dims)

        # actor's layer
        self.action_head = nn.Linear(h_dims, action_dims, bias=False)
        # critic's layer
        self.value_head = nn.Linear(h_dims, 1, bias=False)
        
        # init
        torch.nn.init.xavier_normal_(self.affine1.weight)
        torch.nn.init.xavier_normal_(self.affine2.weight)
        torch.nn.init.xavier_normal_(self.action_head.weight)
        torch.nn.init.xavier_normal_(self.value_head.weight)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, inputs):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(inputs))
        x = F.relu(self.affine2(x))

        x = x.view(1, 16)
        hx, cx = self.lstmcell(x, (self.hx, self.cx))
        x = hx
        self.hx = hx.detach()
        self.cx = cx.detach()

        # actor: choses action to take from state s_t 
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t 
        return action_prob, state_values
