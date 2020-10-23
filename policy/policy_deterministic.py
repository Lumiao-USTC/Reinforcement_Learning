import torch
import numpy as np
from policy.policy import basicPolicy
from torchs.networks.neural_network import neuralNetwork


class deterministicPolicy(basicPolicy):
    def __init__(self, action_space, observation_space, network_params):
        self.action_space = action_space
        self.observation_space = observation_space
        self.neural_network = neuralNetwork(**network_params)

    def get_action(self, observation_numpy):
        """
        :param observation_numpy: this is an ndarray object and should not have the dimension of the sample.
        :return: this is also an ndarray object.
        """
        return (self.get_action_torch(torch.from_numpy(np.array([observation_numpy])).float()).detach().numpy())[0], {}

    def get_action_torch(self, observation_tensor):
        """
        :param observation_tensor: this is a tensor object and should have the dimension of the sample.
        :return: this is also a tensor object.
        """
        return self.neural_network.forward(observation_tensor)

    def reset(self):
        pass


class deterministicPolicyWithGaussianNoise:
    def __init__(self, deterministic_policy, sigma, clip):
        self.policy = deterministic_policy
        self.sigma = sigma
        self.clip = clip

    def get_action(self, observation):
        # The action should be bounded in the same way as the original policy!
        action, policy_info = self.policy.get_action(observation)
        action_noise = action + np.clip(np.random.normal(0, self.sigma, action.shape), -self.clip, self.clip)
        return action_noise, policy_info

    def reset(self):
        pass
