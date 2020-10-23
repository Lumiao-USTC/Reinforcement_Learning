from environment.environment_maker import environment_maker
from utils.variants import network_architecture
from torchs.networks.neural_network import neuralNetworkMultiInput
from torchs.torch_utils.torch_utils_networks import update_one_from_the_other
from policy.policy_deterministic import deterministicPolicy, deterministicPolicyWithGaussianNoise
from collector.replay_buffer.replay_buffer_from_environment import replayBufferFromEnv
from collector.data_collector.data_collector_path import dataCollectorPath
from collector.data_collector.data_collector_steps import dataCollectorSteps
from torch.nn import MSELoss
from torch import optim
from trainer.examples.td3_trainer import td3Trainer


class td3Agent(object):
    def __init__(self,
                 environment_id,
                 agent_parameters,
                 load_agent_path=None
                 ):
        self.environment_id = environment_id
        self.environment_evaluation_id = environment_id + "_evaluation"
        self.environment_exploration_id = environment_id + "_exploration"
        self.environment_evaluation = environment_maker(self.environment_evaluation_id)
        self.environment_exploration = environment_maker(self.environment_exploration_id)

        self.online_q_function_1 = neuralNetworkMultiInput(
            **network_architecture(self.environment_id, 'td3_q_function')
        )
        self.target_q_function_1 = neuralNetworkMultiInput(
            **network_architecture(self.environment_id, 'td3_q_function')
        )
        update_one_from_the_other(
            self.online_q_function_1, self.target_q_function_1, 1.0
        )
        self.online_q_function_2 = neuralNetworkMultiInput(
            **network_architecture(self.environment_id, 'td3_q_function')
        )
        self.target_q_function_2 = neuralNetworkMultiInput(
            **network_architecture(self.environment_id, 'td3_q_function')
        )
        update_one_from_the_other(
            self.online_q_function_2, self.target_q_function_2, 1.0
        )

        self.evaluation_policy = deterministicPolicy(
            self.environment_evaluation.action_space.low.size,
            self.environment_evaluation.observation_space.low.size,
            network_architecture(self.environment_id, 'td3_policy')
        )
        self.target_evaluation_policy = deterministicPolicy(
            self.environment_evaluation.action_space.low.size,
            self.environment_evaluation.observation_space.low.size,
            network_architecture(self.environment_id, 'td3_policy')
        )
        update_one_from_the_other(
            self.evaluation_policy.neural_network, self.target_evaluation_policy.neural_network, 1.0
        )
        self.exploration_policy = deterministicPolicyWithGaussianNoise(
            self.evaluation_policy, 0.1, 2
        )

        self.replay_buffer = replayBufferFromEnv(
            agent_parameters['replay_buffer_size'], self.environment_exploration
        )
        self.evaluation_collector = dataCollectorPath(
            self.environment_evaluation, self.evaluation_policy, render=True
        )
        self.exploration_collector = dataCollectorSteps(
            self.environment_exploration, self.exploration_policy, render=True
        )

        self.loss_criterion = MSELoss()
        self.optimizer = optim.Adam
        self.trainer = td3Trainer(
            self.online_q_function_1,
            self.online_q_function_2,
            self.target_q_function_1,
            self.target_q_function_2,
            self.evaluation_policy,
            self.target_evaluation_policy,
            self.loss_criterion,
            self.optimizer,
            **agent_parameters['trainer_parameters']
        )
        self.discount_factor = agent_parameters['discount_factor']
