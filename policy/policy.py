import abc


class basicPolicy(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_action(self, observation, **kwargs):
        pass

    def reset(self):
        pass
