import abc


class basic_algorithm(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 agent,
                 num_exploration_step_before_first_epoch,
                 start_epoch,
                 num_epoch,
                 num_train_loop_per_epoch
                 ):
        self.agent = agent
        self.num_exploration_step_before_first_epoch = num_exploration_step_before_first_epoch
        self.start_epoch = start_epoch
        self.num_epoch = num_epoch
        self.num_train_loop_per_epoch = num_train_loop_per_epoch

    @abc.abstractmethod
    def going_through_epochs_and_training(self):
        pass

    def exploration_before_first_epoch(self):
        pass

    @abc.abstractmethod
    def end_epoch(self, current_epoch):
        """
        Saving data generated in the epoch, initiating new collectors
        Updating agent's hyperparameters
        """
        pass

    @abc.abstractmethod
    def train_mode(self, mode):
        pass

