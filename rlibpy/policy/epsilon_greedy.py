from .base_policy import BasePolicy

import numpy as np


def get_decay_rate(epsilon_start, epsilon_end, num_episodes):
    return (epsilon_end / epsilon_start) ** (1.0 / (num_episodes - 1))


class EpsilonGreedy(BasePolicy):
    def __init__(self, epsilon, decay_rate=1):
        self._epsilon_start = epsilon
        self.epsilon = epsilon
        self.decay_rate = decay_rate

    def choose(self, values, evaluate=False):
        if np.random.rand() > self.epsilon or evaluate:
            return np.argmax(values)
        else:
            return np.random.randint(values.size)

    def after_episode(self):
        self.epsilon *= self.decay_rate

    def reset(self):
        self.epsilon = self._epsilon_start
