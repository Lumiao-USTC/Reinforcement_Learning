import random
import torch
import numpy as np
from policy.policy import basicPolicy


class epsilonGreedyPolicy(basicPolicy):
    def __init__(self, q_function, action_space, epsilon):
        self.q_function = q_function
        self.action_space = action_space
        self.epsilon = epsilon

    def get_action(self, observation):
        """
        :param observation: The observation shouldn't have the dimension of sample.
        :return: action, dictionary
        """
        action_max = self.q_function.forward(torch.from_numpy(np.array([observation])).float()).max(1)[1]
        if random.random() <= self.epsilon:
            return self.action_space.sample(), {}
        return int(action_max.detach().numpy()), {}

    def set_epsilon(self, epsilon_new):
        self.epsilon = epsilon_new
