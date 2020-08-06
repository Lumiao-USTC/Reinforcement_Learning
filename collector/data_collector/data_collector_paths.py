from collections import deque
from collector.data_collector.data_collector import basic_data_collector
from collector.interactor import interact


class dataCollectorPaths(basic_data_collector):
    def __init__(self,
                 environment,
                 policy,
                 size_of_collector=None,
                 render=True,
                 render_kwargs=None
                 ):
        self.environment = environment
        self.policy = policy
        self.size_of_collector = size_of_collector
        self.collector = deque(maxlen=self.size_of_collector)
        self.num_paths_in_collector = 0
        self.num_steps_in_collector = 0
        self.render = render
        self.render_kwargs = render_kwargs

    def collect_data(self, num_collect_steps, max_path_length):
        """
        This function collects path data with given step numbers. The return will be a list of steps. Also,
        the list will be append to the collector.
        :param num_collect_steps: number of steps to be collected.
        :param max_path_length: w.r.t. function "interact", which sets the maximal length of a single path.
        """
        paths = []
        current_steps = 0
        while current_steps < num_collect_steps:
            path = interact(
                self.environment,
                self.policy,
                min(num_collect_steps - current_steps, max_path_length),
                self.render,
                self.render_kwargs
            )
            current_steps += len(path['actions'])
            paths.append(path)
        self.collector.extend(paths)
        self.num_paths_in_collector += len(paths)
        self.num_steps_in_collector += current_steps

    def end_collect(self):
        self.collector = deque(maxlen=self.size_of_collector)
        self.num_steps_in_collector = 0
        self.num_paths_in_collector = 0

    def get_data(self, include_policy_infos=False):
        if not include_policy_infos:
            data_without_policy_infos = list()
            for _ in range(self.num_paths_in_collector):
                data_without_policy_infos.append(
                    dict(
                        observations=self.collector[_]['observations'],
                        actions=self.collector[_]['actions'],
                        rewards=self.collector[_]['rewards'],
                        next_observations=self.collector[_]['next_observations'],
                        terminals=self.collector[_]['terminals'],
                        environment_infos=self.collector[_]['environment_infos']
                    )
                )
            return data_without_policy_infos
        return self.collector

















