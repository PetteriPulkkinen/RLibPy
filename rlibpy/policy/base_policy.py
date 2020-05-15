class BasePolicy(object):
    def __init__(self):
        pass

    def choose(self, action_values, observation, evaluate=False):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def update(self, observation, action):
        pass

    def after_step(self):
        pass

    def after_episode(self):
        pass
