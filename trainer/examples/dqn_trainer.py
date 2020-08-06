import torch
import numpy as np
from trainer.trainer import basic_trainer
from torchs.convolution_neural_network import cNN


class dqnTrainer(basic_trainer):
    def __init__(self,
                 online_q_function,
                 target_q_function,
                 loss_criterion,
                 optimizer,
                 discount_factor,
                 learning_rate,
                 target_q_update_period,
                 target_update_tau,
                 ):
        self.online_q_function = online_q_function
        self.target_q_function = target_q_function
        self.loss_criterion = loss_criterion
        self.optimizer = optimizer
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.target_q_update_period = target_q_update_period
        self.target_update_tau = target_update_tau

        self.current_train_step = 0

    def train(self, train_data):
        observations = train_data['observations']
        actions = train_data['actions']
        rewards = train_data['rewards']
        next_observations = train_data['next_observations']
        terminals = train_data['terminals']

        target_q_value = self.target_q_function.forward(
            torch.from_numpy(next_observations).float()).detach().max(1, keepdim=True)[0]
        y_label = torch.from_numpy(rewards).detach() + \
                  torch.tensor(self.discount_factor).detach() * torch.from_numpy(1-terminals).detach() * target_q_value
        y_predict = torch.sum(self.online_q_function.forward(
            torch.from_numpy(observations).float()) * torch.from_numpy(actions).detach(), 1, keepdim=True)
        loss = self.loss_criterion.forward(y_predict, y_label)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # for params in self.online_q_function.parameters():
        #     print(params)
        self.current_train_step += 1

        # update target_q_function
        if self.current_train_step % self.target_q_update_period == 0:
            for params_online, params_target in zip(
                self.online_q_function.parameters(), self.target_q_function.parameters()
            ):
                params_target.data.copy_(self.target_update_tau * params_online.data +
                                         (1-self.target_update_tau) * params_target)
