from gym.spaces import Discrete, Box, Tuple


def get_dimension(space):
    if isinstance(space, Box):
        return space.shape
    elif isinstance(space, Discrete):
        return space.n
    elif isinstance(space, Tuple):
        return sum(get_dimension(subspace) for subspace in space.spaces)
    elif hasattr(space, 'flat_dim'):
        return space.flat_dim
    else:
        raise TypeError("Unknown space: {}".format(space))