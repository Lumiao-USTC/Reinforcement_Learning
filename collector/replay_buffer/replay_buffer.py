import abc


class basic_replay_buffer(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def add_step_sample(self,
                        observation,
                        action,
                        reward,
                        next_observation,
                        terminal,
                        environment_info):
        """
        add one transition step (6 parts)
        """
        pass

    def add_batch_step_sample(self, steps):
        """
        add a batch of transition steps
        """
        pass

    def add_paths(self, paths):
        """
        add transition steps from paths
        """
        pass

    @abc.abstractmethod
    def get_sample(self, batch_size):
        pass
