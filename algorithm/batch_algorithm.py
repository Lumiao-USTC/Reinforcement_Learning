import abc
import gtimer as gt
from algorithm.algorithm import basic_algorithm
from collector.data_collector.data_collector_paths import data_collector_paths
from collector.replay_buffer.replay_buffer_from_environment import replay_buffer_from_env


class batch_algorithm(basic_algorithm):
    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 evaluation_environment,
                 exploration_environment,
                 evaluation_collector: data_collector_paths,
                 exploration_collector: data_collector_paths,
                 replay_buffer: replay_buffer_from_env,
                 num_exploration_step_before_first_epoch,
                 num_epoch,
                 start_epoch,
                 num_evaluation_step_per_epoch,
                 num_train_loop_per_epoch,
                 num_exploration_step_per_train_loop,
                 num_train_step_per_train_loop,
                 batch_size,
                 max_path_length,
                 trainer):
        super().__init__(
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
            trainer
        )
        self.num_exploration_step_per_train_loop = num_exploration_step_per_train_loop
        self.num_train_step_per_train_loop = num_train_step_per_train_loop
        self.batch_size = batch_size
        self.max_path_length = max_path_length

    def exploration_before_first_epoch(self):
        if self.num_exploration_step_before_first_epoch > 0:
            self.exploration_collector.collect_data(
                self.num_exploration_step_before_first_epoch,
                self.max_path_length
                )
            self.replay_buffer.add_path(self.exploration_collector.get_data(include_policy_infos=False))
            self.exploration_collector.end_colloect()

    def going_through_epochs_and_training(self):
        self.exploration_before_first_epoch()

        for epoch in gt.get_times(range(self.start_epoch, self.num_epoch)):
            self.evaluation_collector.collect_data(
                self.num_evaluation_step_per_epoch,
                self.max_path_length
            )
            for train_loop in self.num_train_loop_per_epoch:
                self.exploration_collector.collect_data(
                    self.num_exploration_step_per_train_loop,
                    self.max_path_length
                )
                self.replay_buffer.add_path(self.exploration_collector.get_data(include_policy_infos=False))

                self.train_mode(True)
                for train_step in self.num_train_step_per_train_loop:
                    train_data = self.replay_buffer.random_sample(self.batch_size)
                    self.trainer.train(train_data)
                self.train_mode(False)
            self.end_epoch()










