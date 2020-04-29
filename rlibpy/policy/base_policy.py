class BasePolicy(object):
    def __init__(self):
        pass

    def choose(self, values, evaluate=False):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def after_step(self):
        pass

    def after_episode(self):
        pass
