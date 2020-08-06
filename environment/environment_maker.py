import gym
import environment.atari as atari


def environment_maker(environment_id):
    if environment_id == "Riverraid":
        env = atari.make_atari("RiverraidNoFrameskip-v4")
        env = atari.wrap_deepmind(env, episode_life=False, clip_rewards=True, frame_stack=True)
        return env
