import abc
import gtimer as gt
from algorithm.algorithm import basic_algorithm
from collector.data_collector.data_collector_paths import dataCollectorPaths
from collector.interactor import interact


class batch_algorithm(basic_algorithm):
    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 evaluation_environment,
                 exploration_environment,
                 evaluation_collector: dataCollectorPaths,
                 exploration_collector: dataCollectorPaths,
                 replay_buffer,
                 trainer,
                 start_epoch,
                 num_exploration_step_before_first_epoch,
                 num_epoch,
                 num_evaluation_step_per_epoch,
                 num_train_loop_per_epoch,
                 num_exploration_step_per_train_loop,
                 num_train_step_per_train_loop,
                 batch_size,
                 max_path_length,
                 discount_factor
                 ):
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
        self.discount_factor = discount_factor

    def exploration_before_first_epoch(self):
        if self.num_exploration_step_before_first_epoch > 0:
            print("--explore before first epoch--")
            self.exploration_collector.collect_data(
                self.num_exploration_step_before_first_epoch,
                self.max_path_length
                )
            self.replay_buffer.add_paths(self.exploration_collector.get_data(include_policy_infos=False))
            self.exploration_collector.end_collect()

    def evaluation(self):
        self.evaluation_collector.collect_data()
        print("total reward: %f" % self.evaluation_collector.get_path_reward(self.discount_factor))

    def going_through_epochs_and_training(self):
        self.exploration_before_first_epoch()

        for epoch in gt.timed_for(range(self.start_epoch, self.num_epoch)):
            print("--epoch %d--" % (epoch+1))
            print("--epoch %d, evaluation--" % (epoch+1))
            self.evaluation()

            print("--epoch %d, exploration--" % (epoch+1))
            for train_loop in range(self.num_train_loop_per_epoch):
                print("--epoch %d, exploration, trainloop %d--" % (epoch+1, train_loop+1))
                self.exploration_collector.collect_data(
                    self.num_exploration_step_per_train_loop,
                    self.max_path_length
                )
                print("--add paths to replay buffer--")
                self.replay_buffer.add_paths(self.exploration_collector.get_data(include_policy_infos=False))

                self.train_mode(True)
                for train_step in range(self.num_train_step_per_train_loop):
                    print("--epoch %d, exploration, trainloop %d, trainstep %d--" %
                          (epoch+1, train_loop+1, train_step+1))
                    train_data = self.replay_buffer.get_sample(self.batch_size)
                    self.trainer.train(train_data)
                self.train_mode(False)
            self.end_epoch()

    def end_epoch(self):
        self.exploration_collector.end_collect()









