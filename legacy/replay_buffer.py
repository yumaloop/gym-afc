import numpy as np


class ReplayBuffer:
    def __init__(self, state_shape=(100, 100, 3), action_dims=2, max_size=100000):
        h, w, c = state_shape
        self.current_states = np.empty((0, h, w, c), dtype=np.float64)
        self.next_states = np.empty((0, h, w, c), dtype=np.float64)
        self.actions = np.empty((0, action_dims), dtype=np.float64)
        self.rewards = np.empty((0, 1), dtype=np.float64)
        self.ends = np.empty((0, 1), dtype=np.float64)
        self.total_size = 0
        self.max_size = max_size

    def store(self, current_state, action, reward, next_state, end):
        if self.total_size < self.max_size:
            self.rewards = np.append(self.rewards, np.array([[reward]]), axis=0)
            self.ends = np.append(self.ends, np.array([[end]]), axis=0)
            self.actions = np.append(self.actions, np.array([action]), axis=0)
            self.current_states = np.append(
                self.current_states, np.array([current_state]), axis=0
            )
            self.next_states = np.append(
                self.next_states, np.array([next_state]), axis=0
            )

            self.total_size += 1

    def fetch_sample(self, num_samples):

        if num_samples > self.total_size:
            num_samples = self.total_size

        idx = np.random.choice(
            range(min(self.total_size, self.max_size)), size=num_samples, replace=False
        )

        current_states_ = self.current_states[idx]
        actions_ = self.actions[idx]
        rewards_ = self.rewards[idx]
        next_states_ = self.next_states[idx]
        ends_ = self.ends[idx]

        return current_states_, actions_, rewards_, next_states_, ends_
