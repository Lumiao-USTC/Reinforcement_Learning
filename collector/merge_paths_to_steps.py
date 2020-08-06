import numpy as np


def merge_paths_to_steps(paths, num_paths, include_policy_infos):
    steps = {}
    if num_paths == 0:
        return steps
    observations = paths[0]['observations']
    actions = paths[0]['actions']
    rewards = paths[0]['rewards']
    next_observations = paths[0]['next_observations']
    terminals = paths[0]['terminals']
    if include_policy_infos:
        policy_infos = paths[0]['policy_infos']
    environment_infos = paths[0]['environment_infos']

    for _ in range(1, num_paths):
        observations = np.vstack((observations, paths[_]['observations']))
        actions = np.vstack((actions, paths[_]['actions']))
        rewards = np.vstack((rewards, paths[_]['rewards']))
        next_observations = np.vstack((next_observations,
                                       paths[_]['next_observations']))
        terminals = np.vstack((terminals, paths[_]['terminals']))
        if include_policy_infos:
            policy_infos += paths[_]['policy_infos']
        environment_infos += paths[_]['environment_infos']

    steps['observations'] = observations
    steps['actions'] = actions
    steps['rewards'] = rewards
    steps['next_observations'] = observations
    steps['terminals'] = terminals
    if include_policy_infos:
        steps['policy_infos'] = policy_infos
    steps['environment_infos'] = environment_infos
    return steps

