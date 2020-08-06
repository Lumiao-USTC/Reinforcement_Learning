import numpy as np
from collector.data_collector.data_collector import basic_data_collector
from collector.interactor import interact


class dataCollectorPath(basic_data_collector):
    def __init__(self,
                 environment,
                 policy,
                 render=False,
                 render_kwargs=None
                 ):
        self.environment = environment
        self.policy = policy
        self.path = {}
        self.render = render
        self.render_kwargs = render_kwargs

    def collect_data(self):
        self.path = interact(self.environment, self.policy, render=True)

    def end_collect(self):
        self.path = {}

    def get_data(self, include_policy_infos=False):
        if not include_policy_infos:
            return dict(
                    observations=self.path['observations'],
                    actions=self.path['actions'],
                    rewards=self.path['rewards'],
                    next_observations=self.path['next_observations'],
                    terminals=self.path['terminals'],
                    environment_infos=self.path['environment_infos']
                    )
        return self.path

    def get_path_reward(self, discount_factor):
        reward = 0
        path_length = len(self.path['actions'])
        print(path_length)
        for _ in range(path_length):
            reward += self.path['rewards'][_] * np.power(discount_factor, _)
        return reward
