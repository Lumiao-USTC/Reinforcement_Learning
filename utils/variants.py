def network_architecture(environment_id, model):
    if environment_id == 'Riverraid' and model == 'dqn':
        return dict(
            input_width=84,
            input_height=84,
            input_channels=4,
            output_size=18,
            kernel_sizes=[8, 4, 3],
            n_channels=[32, 64, 64],
            strides=[4, 2, 1],
            paddings=[0, 0, 0],
            hidden_sizes=[512],
            batch_norm_conv=False,
            batch_norm_fc=False
        )


def trainer_parameters(trainer_id):
    if trainer_id == 'dqn_trainer':
        return dict(
            discount_factor=0.99,
            learning_rate=0.01,
            target_q_update_period=1,
            target_update_tau=2e-3
        )


def algorithm_parameters(algorithm_id):
    if algorithm_id == 'batch_algorithm':
        return dict(
            num_exploration_step_before_first_epoch=1000,
            num_epoch=3000,
            num_evaluation_step_per_epoch=1000,
            num_train_loop_per_epoch=1,
            num_exploration_step_per_train_loop=2000,
            num_train_step_per_train_loop=200,
            batch_size=16,
            max_path_length=1000,
            discount_factor=0.99
        )

