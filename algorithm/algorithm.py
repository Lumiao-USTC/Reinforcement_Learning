import abc


class basic_algorithm(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 evaluation_environment,
                 exploration_environment,
                 evaluation_collector,
                 exploration_collector,
                 replay_buffer,
                 num_exploration_step_before_first_epoch,
                 num_epoch,
                 start_epoch,
                 num_evaluation_step_per_epoch,
                 num_train_loop_per_epoch,
                 trainer):
        self.evaluation_environment = evaluation_environment
        self.exploration_environment = exploration_environment
        self.evaluation_collector = evaluation_collector
        self.exploration_collector = exploration_collector
        self.replay_buffer = replay_buffer
        self.num_exploration_step_before_first_epoch = num_exploration_step_before_first_epoch
        self.num_epoch = num_epoch
        self.num_evaluation_step_per_epoch = num_evaluation_step_per_epoch
        self.num_train_loop_per_epoch = num_train_loop_per_epoch
        self.trainer = trainer
        self.start_epoch = start_epoch

    @abc.abstractmethod
    def going_through_epochs_and_training(self):
        pass

    def exploration_before_first_epoch(self):
        pass

    @abc.abstractmethod
    def end_epoch(self):
        """
        Saving data generated in the epoch, then initiate new collectors
        """
        pass

    @abc.abstractmethod
    def train_mode(self, mode):
        pass

