"""
https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import collections
import gym
import numpy as np
import statistics
import tensorflow as tf
import tqdm

from matplotlib import pyplot as plt
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple

from environment import AfcEnv

# Create the environment
# env = gym.make("CartPole-v0")
env = AfcEnv()

# Set seed for experiment reproducibility
seed = 42
env.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()


class ActorCritic(tf.keras.Model):
    """Combined actor-critic network."""

    def __init__(self, input_shape: tuple, num_actions: int, num_hidden_units: int):
        """Initialize."""
        super().__init__()

        # w_init = tf.random_normal_initializer(0.0, 0.01)
        w_init = tf.keras.initializers.GlorotNormal()

        self.conv1 = tf.keras.layers.Conv2D(64, 5, activation="relu", kernel_initializer=w_init, input_shape=input_shape)
        self.conv2 = tf.keras.layers.Conv2D(64, 5, activation="relu", kernel_initializer=w_init)
        self.dense1 = tf.keras.layers.Dense(num_hidden_units, activation="relu", kernel_initializer=w_init)
        self.dense2 = tf.keras.layers.Dense(num_hidden_units, activation="relu", kernel_initializer=w_init)
        self.flatten = tf.keras.layers.Flatten()
        self.bn1 = tf.keras.layers.BatchNormalization(axis=1)
        self.bn2 = tf.keras.layers.BatchNormalization(axis=-1)
        
        self.actor_mu = tf.keras.layers.Dense(num_actions, activation="tanh", kernel_initializer=w_init) # estimated action value # mu: -1 ~ 1
        self.actor_sigma = tf.keras.layers.Dense(num_actions, activation="softmax", kernel_initializer=w_init) # estimated action value # sd:  0 ~ 1
        self.critic = tf.keras.layers.Dense(1)

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.bn1(x)
        x = self.dense1(x)
        x = self.bn2(x)
        x = self.dense2(x)

        a_mu = self.actor_mu(x)
        a_sigma = self.actor_sigma(x)
        value = self.critic(x)
        return a_mu, a_sigma, value

num_actions = 3
num_hidden_units = 128
input_shape = (100, 100, 3)
# input_shape = (512, 683, 3)
model = ActorCritic(input_shape, num_actions, num_hidden_units)


def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns state, reward and done flag given an action."""
    state, reward, done, _ = env.step(action)
    return (
        state.astype(np.float32),
        np.array(reward, np.float32),
        np.array(done, np.int32),
    )


def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
    return tf.numpy_function(env_step, [action], [tf.float32, tf.float32, tf.int32])


def run_episode(
    initial_state: tf.Tensor, model: tf.keras.Model, max_steps: int
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Runs a single episode to collect training data."""

    # action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    actions = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    initial_state_shape = initial_state.shape
    state = initial_state

    for t in tf.range(max_steps):
        # Convert state into a batched tensor (batch size = 1)
        state = tf.expand_dims(state, 0)

        # Run the model and to get action probabilities and critic value
        # action_logits_t, value = model(state)
        a_mu, a_sigma, value = model(state)
        a_sigma = a_sigma + 0.01
        # tf.print("value:", value)

        # Sample next action from the action probability distribution
        action_rand = tf.random.normal([1], a_mu, a_sigma, tf.float32)
        action_probs_t = tf.compat.v1.distributions.Normal(a_mu, a_sigma).prob(action_rand)
        action = tf.math.tanh(action_rand) # R -> [-1,1]
        """
        tf.print("a_mu:", a_mu)
        tf.print("a_sigma:", a_sigma)
        tf.print("action_rand:", action_rand)
        tf.print("action_probs_t:", action_probs_t)
        """

        # Store critic values
        values = values.write(t, tf.squeeze(value))

        # Store log probability of the action chosen
        actions = actions.write(t, action)
        action_probs = action_probs.write(t, action_probs_t)

        # Apply action to the environment to get next state and reward
        state, reward, done = tf_env_step(action)
        state.set_shape(initial_state_shape)

        # Store reward
        rewards = rewards.write(t, reward)

        if tf.cast(done, tf.bool):
            break


    actions = actions.stack() # list of action-mean
    action_probs = action_probs.stack() # list of action-sigma
    values = values.stack()
    rewards = rewards.stack()

    return actions, action_probs, values, rewards


def get_expected_return(
    rewards: tf.Tensor, gamma: float, standardize: bool = True
) -> tf.Tensor:
    """Compute expected returns per timestep."""

    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)

    # Start from the end of `rewards` and accumulate reward sums
    # into the `returns` array
    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    discounted_sum = tf.constant(0.0)
    discounted_sum_shape = discounted_sum.shape
    for i in tf.range(n):
        reward = rewards[i]
        discounted_sum = reward + gamma * discounted_sum
        discounted_sum.set_shape(discounted_sum_shape)
        returns = returns.write(i, discounted_sum)
    returns = returns.stack()[::-1]

    if standardize:
        returns = (returns - tf.math.reduce_mean(returns)) / (
            tf.math.reduce_std(returns) + eps
        )

    return returns


huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)


def compute_loss(
        action_probs: tf.Tensor, values: tf.Tensor, returns: tf.Tensor
) -> tf.Tensor:
    """Computes the combined actor-critic loss."""

    advantage = returns - values
    td = tf.subtract(returns, values)

    # actor 
    # action_log_probs = tf.math.log(action_probs)
    # actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)
    action_log_probs = tf.math.log(action_probs)
    actor_loss = -tf.math.reduce_mean(action_log_probs * td)

    # critic
    # td = tf.subtract(returns, values)
    # critic_loss = tf.reduce_mean(tf.square(td))
    critic_loss = huber_loss(values, returns)

    tf.print("a_loss:", actor_loss, "c_loss:", critic_loss)

    return actor_loss + critic_loss


optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


@tf.function
def train_step(
    initial_state: tf.Tensor,
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    gamma: float,
    max_steps_per_episode: int,
) -> tf.Tensor:
    """Runs a model training step."""

    with tf.GradientTape() as tape:

        # Run the model for one episode to collect training data
        actions, action_probs, values, rewards = run_episode(
            initial_state, model, max_steps_per_episode
        )

        # Calculate expected returns
        returns = get_expected_return(rewards, gamma)

        # Convert training data to appropriate TF tensor shapes
        action_probs, values, returns = [
            tf.expand_dims(x, 1) for x in [action_probs, values, returns]
        ]

        # Calculating loss values to update our network
        loss = compute_loss(action_probs, values, returns)
    
    # Compute the gradients from the loss
    grads = tape.gradient(loss, model.trainable_variables)

    # Apply the gradients to the model's parameters
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    episode_reward = tf.math.reduce_sum(rewards)

    return episode_reward


min_episodes_criterion = 100
max_episodes = 10000
max_steps_per_episode = 1000

# Cartpole-v0 is considered solved if average reward is >= 195 over 100
# consecutive trials
reward_threshold = 195
running_reward = 0

# Discount factor for future rewards
gamma = 0.9 # 0.99

# Keep last episodes reward
episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

with tqdm.trange(max_episodes) as t:
    for i in t:
        print(f"i={i} start")
        initial_state = tf.constant(env.reset(), dtype=tf.float32)
        print("env.fixs:", env.fixs)
        episode_reward = int(
            train_step(initial_state, model, optimizer, gamma, max_steps_per_episode)
        )
        print("i", i, "fin.")

        episodes_reward.append(episode_reward)
        running_reward = statistics.mean(episodes_reward)

        t.set_description(f"Episode {i}")
        t.set_postfix(episode_reward=episode_reward, running_reward=running_reward)

        # Show average episode reward every 10 episodes
        if i % 10 == 0:
            # pass
            print(f'Episode {i}: running reward: {running_reward}')

        if running_reward > reward_threshold and i >= min_episodes_criterion:
            print("BREAK !!!!")
            break

print(f"\nSolved at episode {i}: average reward: {running_reward:.2f}!")
