from .base_agent import BaseAgent
from ..policy.base_policy import BasePolicy

import gym
import numpy as np


class QLearningAgent(BaseAgent):
    def __init__(self, environment: gym.Env, policy: BasePolicy, alpha: float, gamma: float, debug=False):
        super().__init__(environment, policy, debug)
        self.environment = environment
        self.policy = policy
        self.alpha = alpha
        self.gamma = gamma
        shape = (environment.observation_space.n, environment.action_space.n)
        self.table = np.zeros(shape=shape, dtype=float)
        self.n_table = np.zeros(shape=shape, dtype=int)

        self.q_hd = np.empty(shape=shape, dtype=object)  # Historical data of Q-values
        self._initialize_covergence_analytics()

    def act(self, observation, evaluate=False):
        values = self.table[observation]
        return self.policy.choose(values, evaluate=evaluate)

    def update(self, action, observation, next_observation, reward):
        next_act = self.act(next_observation, evaluate=True)
        next_q = self.table[next_observation, next_act]
        current_q = self.table[observation, action]
        self.table[observation, action] = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
        self.n_table[observation, action] += 1  # count the updates
        if self.debug:
            self.q_hd[observation, action].append(self.table[observation, action])

    def reset(self):
        self.table = np.zeros_like(self.table)
        self._initialize_covergence_analytics()
        self.n_table = np.zeros_like(self.table, dtype=int)
        self.policy.reset()

    def _initialize_covergence_analytics(self):
        for i in range(self.environment.observation_space.n):
            for j in range(self.environment.action_space.n):
                self.q_hd[i, j] = [0]


class AdaptiveQLearningAgent(QLearningAgent):
    def __init__(self, environment: gym.Env, policy: BasePolicy, omega: float, gamma: float, debug=False):
        super().__init__(environment, policy, alpha=None, gamma=gamma, debug=debug)
        self.omega = omega

    def update(self, action, observation, next_observation, reward):
        next_act = self.act(next_observation, evaluate=True)
        next_q = self.table[next_observation, next_act]
        current_q = self.table[observation, action]
        alpha = 1/((1+self.n_table[observation, action])**self.omega)
        self.table[observation, action] = current_q + alpha * (reward + self.gamma * next_q - current_q)
        self.n_table[observation, action] += 1  # count the updates
        if self.debug:
            self.q_hd[observation, action].append(self.table[observation, action])
