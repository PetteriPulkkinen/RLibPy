import numpy as np
from rlibpy.policy.base_policy import BasePolicy


def get_myopic_exploration_constant(max_reward):
    opt_norm_c = np.sqrt(2)
    return opt_norm_c * max_reward


class UCB(BasePolicy):
    def __init__(self, c: float, n_acts: int, n_obs: int):
        """

        :param c: Exploration constant
        :param n_acts: Number of actions
        :param n_obs: Number of states
        """
        super().__init__()
        self.c = c
        self.t = np.ones(n_obs, dtype=int)
        self.n_table = np.zeros((n_obs, n_acts), dtype=int)

    def choose(self, action_values, observation, evaluate=False):
        if evaluate:
            return np.argmax(action_values)

        n_selected = self.n_table[observation]
        with np.errstate(divide='ignore', invalid='ignore'):
            indexes = action_values + self.c * np.sqrt(np.log(self.t[observation])/n_selected)

        indexes[n_selected < 1] = np.inf
        return np.random.choice(np.nonzero(indexes == np.max(indexes))[0])

    def update(self, observation, action):
        self.n_table[observation, action] += 1
        self.t[observation] += 1

    def reset(self):
        self.t.fill(1)
        self.n_table.fill(0)
