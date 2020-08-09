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


def parameters(agent_id, algorithm_id):
    if agent_id == 'dqn' and algorithm_id == 'batch_algorithm':
        return dict(
            agent_parameters=dict(
                discount_factor=0.99,
                replay_buffer_size=int(1E5),
                epsilon=dict(
                    initial=1,
                    final=0.1,
                    decay=0.995
                ),
                trainer_parameters=dict(
                    discount_factor=0.99,
                    learning_rate=0.001,
                    target_q_update_period=1,
                    target_update_tau=2e-3
                ),
            ),
            algorithm_parameters=dict(
                num_exploration_step_before_first_epoch=1000,
                num_epoch=3000,
                num_train_loop_per_epoch=500,
                num_exploration_step_per_train_loop=1,
                num_train_step_per_train_loop=1,
                batch_size=64,
                max_path_length=1000,
            )
        )
