import gym
import numpy as np
from os import path

from rlibpy.policy.base_policy import BasePolicy


class BaseAgent(object):
    def __init__(self, environment: gym.Env, policy: BasePolicy, debug=False):
        self.debug = debug
        self.environment = environment
        self.policy = policy

    def act(self, observation, evaluate=False):
        raise NotImplementedError

    def update(self, action, observation, next_observation, reward):
        raise NotImplementedError

    def learn(self, n_episodes, max_steps, n_log=1, n_save=None, save_folder=None):
        log_reward_sum = 0
        log_steps = 0
        rewards = np.empty(n_episodes, dtype=object)
        for ep in range(n_episodes):
            rewards[ep] = list()
            observation = self.environment.reset()
            reward_sum = 0
            steps = 0
            tn_log = 0
            for st in range(max_steps):
                action = self.act(observation, evaluate=False)
                next_observation, reward, done, ds = self.environment.step(action)

                if ds['us']:
                    rewards[ep].append(reward)
                else:
                    rewards[ep].append(np.nan)

                self.update(action, observation, next_observation, reward)
                observation = next_observation

                reward_sum += reward
                steps += 1
                tn_log += 1
                self.after_step()

                if done:
                    break
            if n_save is not None:
                if ep % n_save == 0 or ep == n_episodes - 1:
                    self.save(filename=path.join(save_folder, 'state_{}.rla'.format(ep)))
            log_reward_sum += reward_sum
            log_steps += steps
            if ep % n_log == 0 or ep == n_episodes - 1:
                print('episode {}, Avg reward {:.2f}, Avg steps {}'.format(
                    ep+1, log_reward_sum/tn_log, log_steps/tn_log))
                log_reward_sum = 0
                log_steps = 0
            self.after_episode()
        return np.array(rewards)

    def after_step(self):
        self.policy.after_step()

    def after_episode(self):
        self.policy.after_episode()

    def reset(self):
        raise NotImplementedError

    def save(self, filename):
        pass

    def load(self, filename):
        pass

    def visualize(self):
        pass


