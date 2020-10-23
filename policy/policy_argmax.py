import torch
import numpy as np
from policy.policy import basicPolicy


class argmaxPolicy(basicPolicy):
    def __init__(self, q_function):
        self.q_function = q_function

    def get_action(self, observation):
        """
        :param observation: The observation shouldn't have the dimension of sample.
        :return: action, dictionary
        """
        action = self.q_function.forward(torch.from_numpy(np.array([observation])).float()).max(1)[1]
        return int(action.detach().numpy()), {}







