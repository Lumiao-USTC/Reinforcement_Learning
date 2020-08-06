from torch.nn import MSELoss
from torch import optim
from environment.environment_maker import environment_maker
from torchs.convolution_neural_network import cNN
from utils.variants import network_architecture
from policy.policy_argmax import argmaxPolicy
from policy.policy_epsilon_greedy import epsilonGreedyPolicy
from collector.replay_buffer.replay_buffer_from_environment import replayBufferFromEnv
from collector.data_collector.data_collector_paths import dataCollectorPaths
from collector.data_collector.data_collector_path import dataCollectorPath
from trainer.examples.dqn_trainer import dqnTrainer
from algorithm.batch_algorithm import batch_algorithm


class dqnModel(object):
    def __init__(self,
                 environment_id,
                 trainer_parameters,
                 algorithm_parameters
                 ):
        self.environment_id = environment_id
        self.environment = environment_maker(self.environment_id)
        self.online_q_function = cNN(**network_architecture(self.environment_id, 'dqn'))
        self.target_q_function = cNN(**network_architecture(self.environment_id, 'dqn'))
        self.evaluation_policy = argmaxPolicy(self.online_q_function)
        self.exploration_policy = epsilonGreedyPolicy(self.online_q_function, self.environment.action_space, 1)
        self.replay_buffer = replayBufferFromEnv(int(1E5), self.environment)
        self.evaluation_collector = dataCollectorPath(self.environment, self.evaluation_policy)
        self.exploration_collector = dataCollectorPaths(self.environment, self.exploration_policy)
        self.loss_criterion = MSELoss()
        self.optimizer = optim.Adam(
            self.online_q_function.parameters(),
            trainer_parameters['learning_rate']
        )
        self.trainer = dqnTrainer(
            online_q_function=self.online_q_function,
            target_q_function=self.target_q_function,
            loss_criterion=self.loss_criterion,
            optimizer=self.optimizer,
            **trainer_parameters
        )
        self.algorithm = batch_algorithm(
            self.environment,
            self.environment,
            self.evaluation_collector,
            self.exploration_collector,
            self.replay_buffer,
            self.trainer,
            0, **algorithm_parameters
        )


