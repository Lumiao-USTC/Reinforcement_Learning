import random
import torch
import numpy as np
from policy.policy import basicPolicy


class epsilonGreedyPolicy(basicPolicy):
    def __init__(self, q_function, action_space, epsilon: dict):
        self.q_function = q_function
        self.action_space = action_space
        self.epsilon = epsilon
        self.current_epsilon = epsilon['initial']

    def get_action(self, observation):
        """
        :param observation: The observation shouldn't have the dimension of sample.
        :return: action, dictionary
        """
        action_max = self.q_function.forward(torch.from_numpy(np.array([observation])).float()).max(1)[1]
        if random.random() <= self.current_epsilon:
            return self.action_space.sample(), {}
        return int(action_max.detach().numpy()), {}

    def epsilon_decay(self):
        self.current_epsilon = max(self.current_epsilon * self.epsilon['decay'], self.epsilon['final'])
