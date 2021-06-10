import numpy as np
import tensorflow as tf
from environment import AfcEnvPO

model = tf.keras.models.load_model(
    "/root/gym-afc/saved_model/tmp_model_afcpo_episode9900"
)
model.summary()

env = AfcEnvPO()
init_state = env.reset()
state = tf.constant(init_state, dtype=tf.float32)

for i in range(100):
    state = tf.expand_dims(state, 0)

    # state -> action
    a_mu, a_sigma, value = model(state)
    a_sigma = a_sigma + 0.01

    tf.print(i, a_mu, value)
    action_rand = tf.random.normal([1], a_mu, a_sigma, tf.float32)
    action = tf.math.tanh(action_rand)  # R -> [-1,1]
    # action = a_mu

    print(env.fixs)

    # action -> state, reward
    state, _, done, _ = env.step(action)

    print(type(state), np.mean(state))

    state = tf.constant(state, dtype=tf.float32)

    if done:
        break
