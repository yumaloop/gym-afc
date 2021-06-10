"""
https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic
"""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import collections
import gym
import numpy as np
import statistics
import tensorflow as tf
import tqdm

from matplotlib import pyplot as plt
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple

from environment import AfcEnvPO
from sac import SoftActorCritic
from replay_buffer import ReplayBuffer

# Create the environment
env = AfcEnvPO()

# Set seed for experiment reproducibility
seed = 42
env.seed(seed)
tf.random.set_seed(seed)
tf.keras.backend.set_floatx("float32")
np.random.seed(seed)

# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()
action_dims = 2
input_shape = (100, 100, 3)

model_path = "/root/gym-afc/saved_model/model-202102435643"
model_name = "model-12343-213245"
writer = tf.summary.create_file_writer(model_path + model_name + "/summary")
model = SoftActorCritic(action_dims, writer)
replay = ReplayBuffer(input_shape, action_dims)

epochs = 50
batch_size = 128
start_steps = 0
global_step = 1
episode = 1
episode_rewards = []
verbose = True

while True:
    current_state = env.reset()
    current_state = tf.convert_to_tensor(current_state, dtype=tf.float32)

    step = 1
    episode_reward = 0
    done = False

    while not done:
        print("global_step:", global_step, "step:", step)

        if global_step < start_steps:
            if np.random.uniform() > 0.8:
                action = env.action_space.sample()
            else:
                action = model.sample_action(current_state)
        else:
            action = model.sample_action(current_state)


        # Execute action, observe next state and reward
        next_state, reward, done, _ = env.step(action)
        next_state = tf.convert_to_tensor(current_state, dtype=tf.float32)

        episode_reward += reward
        
        tf.print("action:", action, "reward:", reward)

        # Set end to 0 if the episode ends otherwise make it 1
        # although the meaning is opposite but it is just easier to mutiply
        # with reward for the last step.
        if done:
            end = 0
        else:
            end = 1

        # Store transition in replay buffer
        replay.store(current_state, action, reward, next_state, end)

        # Update current state
        current_state = next_state

        step += 1
        global_step += 1

    if (step % 1 == 0) and (global_step > start_steps):
        for epoch in range(epochs):
            # Randomly sample minibatch of transitions from replay buffer
            current_states, actions, rewards, next_states, ends = replay.fetch_sample(
                num_samples=batch_size
            )

            # Perform single step of gradient descent on Q and policy
            # network
            critic1_loss, critic2_loss, actor_loss, alpha_loss = model.train(
                current_states, actions, rewards, next_states, ends
            )
            if verbose:
                print(
                    episode,
                    global_step,
                    epoch,
                    critic1_loss.numpy(),
                    critic2_loss.numpy(),
                    actor_loss.numpy(),
                    episode_reward,
                )

            with writer.as_default():
                tf.summary.scalar("actor_loss", actor_loss, sac.epoch_step)
                tf.summary.scalar("critic1_loss", critic1_loss, sac.epoch_step)
                tf.summary.scalar("critic2_loss", critic2_loss, sac.epoch_step)
                tf.summary.scalar("alpha_loss", alpha_loss, sac.epoch_step)

            sac.epoch_step += 1

            if sac.epoch_step % 1 == 0:
                sac.update_weights()

    if episode % 1 == 0:
        model.policy.save_weights(model_path + model_name + "/model")

    episode_rewards.append(episode_reward)
    episode += 1
    avg_episode_reward = sum(episode_rewards[-100:]) / len(episode_rewards[-100:])

    print(f"Episode {episode} reward: {episode_reward}")
    print(f"{episode} Average episode reward: {avg_episode_reward}")
    with writer.as_default():
        tf.summary.scalar("episode_reward", episode_reward, episode)
        tf.summary.scalar("avg_episode_reward", avg_episode_reward, episode)
