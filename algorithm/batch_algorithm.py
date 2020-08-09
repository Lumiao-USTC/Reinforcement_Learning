import abc
from algorithm.algorithm import basic_algorithm


class batch_algorithm(basic_algorithm):
    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 agent,
                 start_epoch,
                 algorithm_parameters
                 ):
        super().__init__(
            agent,
            start_epoch,
            algorithm_parameters['num_epoch'],
            algorithm_parameters['num_evaluation_step_per_epoch'],
            algorithm_parameters['num_train_loop_per_epoch'],
        )
        self.num_exploration_step_per_train_loop = algorithm_parameters['num_exploration_step_per_train_loop']
        self.num_train_step_per_train_loop = algorithm_parameters['num_train_step_per_train_loop']
        self.batch_size = algorithm_parameters['batch_size']
        self.max_path_length = algorithm_parameters['max_path_length']
        self.current_exploration_step = 0

    def going_through_epochs_and_training(self):
        self.exploration_before_first_epoch()

        print(self.start_epoch, self.num_epoch)
        for epoch in range(self.start_epoch, self.num_epoch):
            print("--epoch %d--" % (epoch+1))
            print("--epoch %d, evaluation--" % (epoch+1))
            self.evaluation()

            print("--epoch %d, exploration--" % (epoch+1))
            for train_loop in range(self.num_train_loop_per_epoch):
                print("--epoch %d, exploration, trainloop %d--" % (epoch+1, train_loop+1))
                self.agent.exploration_collector.collect_data(self.num_exploration_step_per_train_loop)
                self.current_exploration_step += self.num_train_step_per_train_loop

                print("--add steps to replay buffer--")
                self.agent.replay_buffer.add_batch_step_sample(
                    self.agent.exploration_collector.get_data(include_policy_infos=False))

                self.train_mode(True)
                for train_step in range(self.num_train_step_per_train_loop):
                    print("--epoch %d, exploration, trainloop %d, trainstep %d--" %
                          (epoch+1, train_loop+1, train_step+1))
                    train_data = self.agent.replay_buffer.get_sample(self.batch_size)
                    self.agent.trainer.train(train_data)
                self.train_mode(False)
            self.end_epoch(epoch)

    def exploration_before_first_epoch(self):
        if self.num_exploration_step_before_first_epoch > 0:
            print("--explore before first epoch--")
            self.agent.exploration_collector.collect_data(self.num_exploration_step_before_first_epoch)
            print("--add steps to replay buffer--")
            self.agent.replay_buffer.add_batch_step_sample(
                self.agent.exploration_collector.get_data(include_policy_infos=False))
            self.agent.exploration_collector.end_collect()

    def evaluation(self):
        self.agent.evaluation_collector.collect_data()
        print("total reward: %f" % self.agent.evaluation_collector.get_path_reward())

    def end_epoch(self, current_epoch):
        self.agent.exploration_collector.end_collect()
        self.agent.hyperparameters_update(current_epoch)

    def train_mode(self, mode):
        pass
