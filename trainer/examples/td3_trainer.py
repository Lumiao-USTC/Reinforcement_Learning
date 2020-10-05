import torch
from torchs.torch_utils.torch_utils_networks import update_one_from_the_other
from torchs.torch_utils.torch_trainer import torchTrainer


class td3Trainer(torchTrainer):
    def __init__(self,
                 online_q_function_1,
                 online_q_function_2,
                 target_q_function_1,
                 target_q_function_2,
                 evaluation_policy,
                 target_evaluation_policy,
                 loss_criterion,
                 optimizer_class,
                 discount_factor,
                 learning_rate_q_function,
                 learning_rate_policy,
                 target_and_policy_update_period,
                 update_tau,
                 epsilon_sigma,
                 epsilon_clip
                 ):
        self.online_q_function_1 = online_q_function_1
        self.online_q_function_2 = online_q_function_2
        self.target_q_function_1 = target_q_function_1
        self.target_q_function_2 = target_q_function_2
        self.evaluation_policy = evaluation_policy
        self.target_evaluation_policy = target_evaluation_policy
        self.loss_criterion = loss_criterion
        self.discount_factor = discount_factor
        self.learning_rate_q_function = learning_rate_q_function
        self.learning_rate_policy = learning_rate_policy
        self.target_and_policy_update_period = target_and_policy_update_period
        self.update_tau = update_tau
        self.epsilon_sigma = epsilon_sigma
        self.epsilon_clip = epsilon_clip
        self.optimizer_q_1 = optimizer_class(
            online_q_function_1.parameters(),
            learning_rate_q_function
        )
        self.optimizer_q_2 = optimizer_class(
            online_q_function_2.parameters(),
            learning_rate_q_function
        )
        self.optimizer_policy = optimizer_class(
            evaluation_policy.neural_network.parameters(),
            learning_rate_policy
        )
        self.current_train_step = 0

    def torch_train(self, train_data):
        observations = train_data['observations']
        actions = train_data['actions']
        rewards = train_data['rewards']
        next_observations = train_data['next_observations']
        terminals = train_data['terminals']
        num_of_data = observations.shape[0]

        target_actions = self.target_evaluation_policy.get_action_batch(next_observations)
        noise = torch.clamp(
            torch.normal(0, self.epsilon_sigma, size=actions.shape), -self.epsilon_clip, self.epsilon_clip
        )
        target_actions_with_noise = target_actions + noise

        target_q_value_1 = self.target_q_function_1.forward(next_observations, target_actions_with_noise)
        target_q_value_2 = self.target_q_function_2.forward(next_observations, target_actions_with_noise)
        target_q_value = torch.min(target_q_value_1, target_q_value_2)
        target = rewards + (1 - terminals) * self.discount_factor * target_q_value
        target = target.detach()

        predict_q_value_1 = self.online_q_function_1.forward(observations, actions)
        predict_q_value_2 = self.online_q_function_2.forward(observations, actions)
        loss_q_1 = self.loss_criterion(predict_q_value_1, target)
        loss_q_2 = self.loss_criterion(predict_q_value_2, target)

        # update Q_function
        self.optimizer_q_1.zero_grad()
        loss_q_1.backward()
        self.optimizer_q_1.step()

        self.optimizer_q_2.zero_grad()
        loss_q_2.backward()
        self.optimizer_q_2.step()

        # update policy and target networks
        if self.current_train_step % self.target_and_policy_update_period == 0:
            q_value = self.online_q_function_1.forward(
                observations, self.evaluation_policy.get_action_batch(observations))
            loss_policy = -q_value.mean()
            self.optimizer_policy.zero_grad()
            loss_policy.backward()
            self.optimizer_policy.step()

            update_one_from_the_other(self.online_q_function_1, self.target_q_function_1, self.update_tau)
            update_one_from_the_other(self.online_q_function_2, self.target_q_function_2, self.update_tau)
            update_one_from_the_other(
                self.evaluation_policy.neural_network, self.target_evaluation_policy.neural_network, self.update_tau
            )

















