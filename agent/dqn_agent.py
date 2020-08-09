from torch.nn import SmoothL1Loss
from torch import optim
from environment.environment_maker import environment_maker
from torchs.convolution_neural_network import cNN
from utils.variants import network_architecture
from policy.policy_argmax import argmaxPolicy
from policy.policy_epsilon_greedy import epsilonGreedyPolicy
from collector.replay_buffer.replay_buffer_from_environment import replayBufferFromEnv
from collector.data_collector.data_collector_path import dataCollectorPath
from collector.data_collector.data_collector_steps import dataCollectorSteps
from trainer.examples.dqn_trainer import dqnTrainer


class dqnAgent(object):
    def __init__(self,
                 environment_id,
                 agent_parameters
                 ):
        self.environment_id = environment_id
        self.environment_evaluation_id = environment_id + "_evaluation"
        self.environment_exploration_id = environment_id + "_exploration"
        self.environment_evaluation = environment_maker(self.environment_evaluation_id)
        self.environment_exploration = environment_maker(self.environment_exploration_id)
        self.online_q_function = cNN(**network_architecture(self.environment_id, 'dqn'))
        self.target_q_function = cNN(**network_architecture(self.environment_id, 'dqn'))
        self.evaluation_policy = argmaxPolicy(self.online_q_function)
        self.exploration_policy = epsilonGreedyPolicy(
            self.online_q_function, self.environment_exploration.action_space, agent_parameters['epsilon'])
        self.replay_buffer = replayBufferFromEnv(agent_parameters['replay_buffer_size'], self.environment_exploration)
        self.evaluation_collector = dataCollectorPath(self.environment_evaluation, self.evaluation_policy)
        self.exploration_collector = dataCollectorSteps(
            self.environment_exploration, self.exploration_policy, render=True)
        self.loss_criterion = SmoothL1Loss()
        self.optimizer = optim.Adam(
            self.online_q_function.parameters(),
            agent_parameters['trainer_parameters']['learning_rate']
        )
        self.trainer = dqnTrainer(
            online_q_function=self.online_q_function,
            target_q_function=self.target_q_function,
            loss_criterion=self.loss_criterion,
            optimizer=self.optimizer,
            **agent_parameters['trainer_parameters']
        )
        self.discount_factor = agent_parameters['discount_factor']

    def hyperparameters_update(self, current_epoch):
        if current_epoch % 10:
            self.exploration_policy.epsilon_decay()
