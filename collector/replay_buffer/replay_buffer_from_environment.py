import numpy as np
from gym.spaces import Discrete
from collector.replay_buffer.replay_buffer import basic_replay_buffer
from utils.get_environment_dimension import get_dimension
from collector.merge_paths_to_steps import merge_paths_to_steps


class replayBufferFromEnv(basic_replay_buffer):
    """
    This replay_buffer initiates via the information of given environment. The data added to
    this buffer should be from the same environment.
    """
    def __init__(self,
                 replay_buffer_size,
                 environment
                 ):
        self.replay_buffer_size = replay_buffer_size
        self.environment = environment
        self.observation_space = environment.observation_space
        self.action_space = environment.action_space

        self.observation_dim = get_dimension(self.observation_space)
        self.action_dim = get_dimension(self.action_space)

        self.observations = np.zeros((self.replay_buffer_size,)+self.observation_dim)
        self.actions = np.zeros((self.replay_buffer_size, self.action_dim))
        self.rewards = np.zeros((self.replay_buffer_size, 1))
        self.next_observations = np.zeros((self.replay_buffer_size,)+self.observation_dim)
        self.terminals = np.zeros((self.replay_buffer_size, 1))
        self.environment_infos = list()
        for _ in range(self.replay_buffer_size):
            self.environment_infos.append({})

        self.next_add_position = 0
        self.current_size = 0

    def add_step_sample(self,
                        observation,
                        action,
                        reward,
                        next_observation,
                        terminal,
                        environment_info):
        if isinstance(self.action_space, Discrete):
            new_action = np.zeros(self.action_dim)
            new_action[action] = 1
        else:
            new_action = action
        self.observations[self.next_add_position] = observation
        self.actions[self.next_add_position] = action
        self.rewards[self.next_add_position] = reward
        self.next_observations[self.next_add_position] = next_observation
        self.terminals[self.next_add_position] = terminal
        self.environment_infos[self.next_add_position] = environment_info

        self.update_add_position(1)

    def update_add_position(self, increment):
        self.next_add_position = (self.next_add_position + increment) % self.replay_buffer_size
        if self.current_size < self.replay_buffer_size:
            self.current_size += 1

    def get_sample(self, batch_size):
        index = np.random.randint(0, self.current_size, batch_size)
        environment_infos = []
        for _ in index:
            environment_infos.append(self.environment_infos[_])
        return dict(
            observations=self.observations[index],
            actions=self.actions[index],
            rewards=self.rewards[index],
            next_observations=self.next_observations[index],
            terminals=self.terminals[index],
            environment_infos=environment_infos
        )

    def add_batch_step_sample(self, steps):
        num_steps = len(steps['actions'])
        for _ in range(num_steps):
            self.add_step_sample(steps['observations'][_],
                                 steps['actions'][_],
                                 steps['rewards'][_],
                                 steps['next_observations'][_],
                                 steps['terminals'][_],
                                 steps['environment_infos'][_])

    def add_paths(self, paths):
        num_paths = len(paths)
        steps = merge_paths_to_steps(paths, num_paths, include_policy_infos=False)
        self.add_batch_step_sample(steps)

    def get_information(self):
        return self.next_add_position, self.current_size




