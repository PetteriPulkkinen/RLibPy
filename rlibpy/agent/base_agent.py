import gym
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

    def learn(self, n_episodes, max_steps, n_log=1):
        log_reward_sum = 0
        log_steps = 0
        for ep in range(n_episodes):
            observation = self.environment.reset()
            reward_sum = 0
            steps = 0
            for st in range(max_steps):
                action = self.act(observation, evaluate=False)
                next_observation, reward, done, _ = self.environment.step(action)

                self.update(action, observation, next_observation, reward)
                observation = next_observation

                reward_sum += reward
                steps += 1
                self.after_step()

                if done:
                    break
            log_reward_sum += reward_sum
            log_steps += steps
            if ep % n_log == 0:
                print('episode {}, Avg reward {:.2f}, Avg steps {}'.format(
                    ep+1, log_reward_sum/n_log, log_steps/n_log))
                log_reward_sum = 0
            self.after_episode()

    def after_step(self):
        self.policy.after_step()

    def after_episode(self):
        self.policy.after_episode()

    def reset(self):
        raise NotImplementedError

    def visualize(self):
        pass


