# https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py

import argparse
import gym
import random
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from environment import AfcEnvGrid

# seed = random.randint(0,100)
seed = 42
gamma = 0.99
log_interval = 10
env = AfcEnvGrid()
env.seed(seed)
torch.manual_seed(seed)


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Policy(nn.Module):
    """
    implements both actor and critic in one model
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


model = Policy()
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()


steps_done = 0
def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)
    
    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)

    # and sample an action using the distribution
    action = m.sample()

    # save to action buffer
    el = SavedAction(m.log_prob(action), state_value)
    model.saved_actions.append(el)

    # the action to take (left or right)
    return action.item()


def finish_episode():
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]:
        # calculate the discounted value
        R = r + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss 
        policy_loss = -log_prob * advantage
        policy_losses.append(policy_loss.float())

        # calculate critic (value) loss using L1 smooth loss
        value_loss = F.smooth_l1_loss(value, torch.tensor([R]))
        value_losses.append(value_loss.float())

    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss = loss.double()

    # perform backprop
    loss.backward()
    optimizer.step()

    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]


def main():
    ep_rewards = []
    running_reward = 10

    # run inifinitely many episodes
    # for i_episode in count(1):
    for i_episode in range(2000):

        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0

        # for each episode, only run 9999 steps so that we don't 
        # infinite loop while learning
        for t in range(1, 10000):

            # select action from policy
            action = select_action(state)

            # take the action
            state, reward, done, _ = env.step(action)

            render = False
            if render:
                env.render()

            model.rewards.append(reward)
            ep_reward += reward

            if done:
                break

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # perform backprop
        finish_episode()

        ep_rewards.append(ep_reward)


        # log results
        if i_episode % log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))

        # check if we have "solved" the cart pole problem
        """
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break
        """

    np.save(f'ep_rewards_seed{seed}',np.array(ep_rewards))

if __name__ == '__main__':
    main()


