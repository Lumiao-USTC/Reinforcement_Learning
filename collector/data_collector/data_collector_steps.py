import numpy as np
from collector.data_collector.data_collector import basic_data_collector


class dataCollectorSteps(basic_data_collector):
    def __init__(self,
                 environment,
                 policy,
                 render=False,
                 render_kwargs=None):
        self.environment = environment
        self.policy = policy
        self.render = render
        if render_kwargs is None:
            render_kwargs = {}
        self.render_kwargs = render_kwargs
        self.current_observation = None
        self.current_terminal = False
        self.collector = {}
        self.observations = []
        self.actions = []
        self.rewards = []
        self.next_observations = []
        self.terminals = []
        self.environment_infos = []
        self.policy_infos = []

    def collect_data(self, num_collect_steps):
        for _ in range(num_collect_steps):
            observation, action, reward, next_observation, terminal, environment_info, policy_info = \
                self.collect_one_step_data()
            self.observations.append(observation)
            self.actions.append(action)
            self.rewards.append(reward)
            self.next_observations.append(next_observation)
            self.terminals.append(terminal)
            self.environment_infos.append(environment_info)
            self.policy_infos.append(policy_info)
        self.collector['observations'] = np.array(self.observations)
        self.collector['actions'] = np.array(self.actions).reshape(-1, 1)
        self.collector['rewards'] = np.array(self.rewards).reshape(-1, 1)
        self.collector['next_observations'] = np.array(self.next_observations)
        self.collector['terminals'] = np.array(self.terminals).reshape(-1, 1)
        self.collector['environment_infos'] = self.environment_infos
        self.collector['policy_infos'] = self.policy_infos
        self.renew_cache()

    def get_data(self, include_policy_infos=False):
        if not include_policy_infos:
            return dict(
                observations=self.collector['observations'],
                actions=self.collector['actions'],
                rewards=self.collector['rewards'],
                next_observations=self.collector['next_observations'],
                terminals=self.collector['terminals'],
                environment_infos=self.collector['environment_infos']
            )
        return self.collector

    def collect_one_step_data(self):
        if self.current_observation is None or self.current_terminal is True:
            self.current_observation = self.environment.reset()
        action, policy_info = self.policy.get_action(self.current_observation)
        next_observation, reward, terminal, environment_info = self.environment.step(action)
        if self.render:
            self.environment.render(**self.render_kwargs)
            print('__render__', action, reward, terminal)
        self.current_observation = next_observation
        self.current_terminal = terminal
        return self.current_observation, action, reward, next_observation, terminal, environment_info, policy_info

    def renew_cache(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.next_observations = []
        self.terminals = []
        self.environment_infos = []
        self.policy_infos = []

    def end_collect(self):
        self.collector = {}
