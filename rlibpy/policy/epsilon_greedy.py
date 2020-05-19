from .base_policy import BasePolicy

import numpy as np
import pickle


def get_decay_rate(epsilon_start, epsilon_end, num_episodes):
    return (epsilon_end / epsilon_start) ** (1.0 / (num_episodes - 1))


class EpsilonGreedy(BasePolicy):
    def __init__(self, epsilon, decay_rate=1):
        self._epsilon_start = epsilon
        self.epsilon = epsilon
        self.decay_rate = decay_rate

    def choose(self, action_values, observation, evaluate=False):
        if np.random.rand() > self.epsilon or evaluate:
            return np.argmax(action_values)
        else:
            return np.random.randint(action_values.size)

    def after_episode(self):
        self.epsilon *= self.decay_rate

    def reset(self):
        self.epsilon = self._epsilon_start
        
    def save(self, filename):
        obj = {
            'epsilon': self.epsilon
        }
        
        with open(filename, 'wb') as out_file:
            pickle.dump(obj, out_file)
            
    def load(self, filename):
        with open(filename, 'rb') as in_file:
            obj = pickle.load(in_file)
    
        for key, value in obj.items():
            setattr(self, key, value)
