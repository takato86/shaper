from gym.spaces import Box, Dict
from decimal import Decimal
import numpy as np


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
        value += i_v * n**i
    return value


def decimal_calc(x, y, symbol):
    d_x = Decimal(str(x))
    d_y = Decimal(str(y))
    if symbol == "-":
        return float(d_x - d_y)
    if symbol == "*":
        return float(d_x * d_y)
    if symbol == "+":
        return float(d_x + d_y)
    if symbol == "/":
        return float(d_x / d_y)
    else:
        raise NotImplementedError


def l2_norm_dist(x_arr, y_arr):
    return np.linalg.norm(x_arr - y_arr, ord=2)
