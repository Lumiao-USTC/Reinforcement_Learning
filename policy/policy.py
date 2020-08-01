import abc


class basic_policy(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_action(self, observation):
        """
        Basic policy class
        :param observation
        :return: action, dictionary
        """
        pass

    def reset(self):
        pass