from gym.spaces import Box, Dict


def get_box(space):
    if type(space) is Box:
        return space
    elif type(space) is Dict:
        return space['observation']

