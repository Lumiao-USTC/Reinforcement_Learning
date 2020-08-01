import numpy as np
import time


def interact(
        environment,
        policy,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None
):
    """
    This function realizes the interaction of the agent (with given policy) with the
    environment. The return is a whole interaction path, unless the length is larger
    than given maximal path length.
    """
    observations = []
    actions = []
    rewards = []
    terminals = []
    environment_infos = []
    policy_infos = []
    if render_kwargs is None:
        render_kwargs = {}

    observation = environment.reset()
    policy.reset()
    next_observation = None
    current_path_length = 0

    if render:
        environment.render(**render_kwargs)
        print('__render__')
        #time.sleep()
    while current_path_length < max_path_length:
        action, policy_info = policy.get_action(observation)
        next_observation, reward, terminal, environment_info = environment.step(action)
        observations.append(observation)
        actions.append(action)
        rewards.append(reward)
        terminals.append(terminal)
        policy_infos.append(policy_info)
        environment_infos.append(environment_info)
        current_path_length += 1
        if render:
            environment.render(**render_kwargs)
            print('__render__', reward, terminal, current_path_length)
            #time.sleep(0.2)
        if terminal:
            break
        observation = next_observation

    observations = np.array(observations)
    actions = np.array(actions).reshape(-1, 1)
    rewards = np.array(rewards).reshape(-1, 1)
    terminals = np.array(terminals).reshape(-1, 1)

    next_observation = np.array([next_observation])
    next_observations = np.vstack((observations[1:], next_observation))

    return dict(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminals=terminals,
        policy_infos=policy_infos,
        environment_infos=environment_infos
    )






