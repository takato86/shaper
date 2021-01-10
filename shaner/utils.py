from gym.spaces import Box, Dict


def get_box(space):
    # 2021/1 robotic arm環境にしか対応できていない
    """
    Args:
        space ([gym.spaces]): [description]

    Returns:
        [gym.Box]: [description]
    """
    if type(space) is Box:
        return space
    elif type(space) is Dict:
        return space['observation']

def n_ary2decimal(array, n):
    """
    Transform n-ary number into decimal number

    Args:
        array ([list]): [n-ary numbers. The element is a digit. The larger digit is smaller index.]
        n ([int]): [description]

    Returns:
        [int]: [decimal]
    """
    value = 0
    for i, i_v in enumerate(array):
        value += n**i
    return value
