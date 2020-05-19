from rlibpy.agent.base_agent import BaseAgent
from rlibpy.policy.base_policy import BasePolicy

import gym
import numpy as np


class QLearningAgent(BaseAgent):
    def __init__(self, environment: gym.Env, policy: BasePolicy, gamma: float, alpha=None, omega=None, debug=False):
        """Q-learning algorithm with various different enhancements.

        Enhancement 1:
            By defining omega in [0, 1] the adaptivity of the alpha parameter is enabled. The adaptivity is based on
            calculating empirical mean (when omega=1) of the rewards for each state-action pair. To prevent unwanted
            behaviour alpha should be None to explicitly tell to use adaptive alpha.

        Enhancement 2:
            This class supports at the moment two different exploration strategies. Those strategies are
                * Epsilon greedy exploration, and
                * Upper Confidence bounds
            The latter strategy should be used only when the algorithm is myopic (gamma=0, omega=1).

        :param environment: OpenAI gym environment
        :param policy: Exploration policy
        :param gamma: Discount rate
        :param alpha: Learning rate (default None)
        :param omega: Decay factor for learning rate (1:='calculate empirical mean', 0:='alpha=1') (default None)
        :param debug: Set True if convergence data need to be saved, otherwise False (default False)
        """
        assert (alpha is not None) is not (omega is not None), "Either alpha or omega should be defined, define " \
                                                               "at most one of them, not both"

        super().__init__(environment, policy, debug)
        self.environment = environment
        self.policy = policy

        self.alpha = alpha
        self.omega = omega
        self.gamma = gamma
        shape = (environment.observation_space.n, environment.action_space.n)
        self.table = np.zeros(shape=shape, dtype=float)
        self.n_table = np.zeros(shape=shape, dtype=int)

        self.q_hd = np.empty(shape=shape, dtype=object)  # Historical data of Q-values
        self.t_hd = np.empty(shape=(
            environment.observation_space.n,
            environment.observation_space.n,
            environment.action_space.n), dtype=object)
        self._initialize_covergence_analytics()
        self._initialize_transition_analytics()

    def act(self, observation, evaluate=False):
        values = self.table[observation]
        return self.policy.choose(values, observation, evaluate=evaluate)

    def update(self, action, observation, next_observation, reward):
        next_act = self.act(next_observation, evaluate=True)
        next_q = self.table[next_observation, next_act]
        current_q = self.table[observation, action]

        if self.omega is None:
            alpha = self.alpha
        else:
            alpha = alpha = 1/((1+self.n_table[observation, action])**self.omega)

        self.table[observation, action] = current_q + alpha * (reward + self.gamma * next_q - current_q)
        self.n_table[observation, action] += 1  # count the updates
        if self.debug:
            self.q_hd[observation, action].append(self.table[observation, action])
            self.t_hd[observation, next_observation, action].append(reward)

        self.policy.update(observation=observation, action=action)

    def reset(self):
        self.table.fill(0)
        self._initialize_covergence_analytics()
        self._initialize_transition_analytics()
        self.n_table.fill(0)
        self.policy.reset()

    def _initialize_covergence_analytics(self):
        for i in range(self.environment.observation_space.n):
            for j in range(self.environment.action_space.n):
                self.q_hd[i, j] = [0]

    def _initialize_transition_analytics(self):
        for i in range(self.environment.observation_space.n):
            for j in range(self.environment.observation_space.n):
                for k in range(self.environment.action_space.n):
                    self.t_hd[i, j, k] = list()
