import logging
import numpy as np
import pandas as pd
import math
from shaner.utils import l2_norm_dist

logger = logging.getLogger(__name__)


class AbstractAchiever:
    def __init__(self, _range, n_obs):
        self._range = _range
        self.n_obs = n_obs

    def eval(self, obs, current_state):
        raise NotImplementedError

    def __generate_subgoals(self):
        raise NotImplementedError


class FourroomsAchiever(AbstractAchiever):
    def __init__(self, _range, n_obs, subgoals, **params):
        super().__init__(_range, n_obs)
        self.subgoals = subgoals[0]  # TODO # 2d-ndarray shape(#obs, #subgoals)

    def eval(self, obs, current_state):
        if len(self.subgoals) <= current_state:
            return False
        subgoal = self.subgoals[current_state]
        return obs == subgoal


class PinballAchiever(AbstractAchiever):
    def __init__(self, _range, n_obs, subgoals, **params):
        super().__init__(_range, n_obs)
        self.subgoals = subgoals  # 2d-ndarray shape(#obs, #subgoals)

    def eval(self, obs, current_state):
        if len(self.subgoals) <= current_state:
            return False
        subgoal = np.array(self.subgoals[current_state])
        idxs = np.argwhere(subgoal == subgoal)  # np.nanでない要素を取り出し
        b_in = l2_norm_dist(
            subgoal[idxs].reshape(-1),
            obs[idxs].reshape(-1)
        ) <= self._range
        res = np.all(b_in)
        if res:
            logger.debug("Achieve the subgoal{}".format(current_state))
        return res

    def __generate_subgoals(self):
        df = pd.read_csv(self.subgoal_path)
        subgoals = df.values
        return subgoals


class FetchPickAndPlaceAchiever(AbstractAchiever):
    def __init__(self, _range, n_obs, subgs, **params):
        """initialize

        Args:
            _range (float): the range to judge whether achieving subgoals.
            n_obs (int): the dimension size of observations
            subgs (np.ndarray): subgoal numpy list.
        """
        super().__init__(_range, n_obs)
        if subgs is not None:
            self.subgoals = subgs
        else:
            self.subgoals = self.__generate_subgoals()  # n_obs=25

    def eval(self, obs, current_state):
        if len(self.subgoals) <= current_state:
            return False
        subgoal = np.array(self.subgoals[current_state])
        idxs = np.argwhere(subgoal == subgoal)  # np.nanでない要素を取り出し
        b_lower = subgoal[idxs] - self._range <= obs[idxs]
        b_higher = obs[idxs] <= subgoal[idxs] + self._range
        res = all(b_lower & b_higher)
        if res:
            logger.info("Achieve the subgoal{}".format(current_state))
        return res

    def __generate_subgoals(self):
        # Subgoal1: Objectの絶対座標[x,y,z] = achieved_goal
        # Subgoal2: Objectの絶対座標とArmの位置が同じでアームを閉じている状態。
        subgoal1 = np.full(self.n_obs, np.nan)
        # subgoal1[6:8] = [0, 0]
        subgoal1[6:9] = [0, 0, 0]
        subgoal2 = np.full(self.n_obs, np.nan)
        # subgoal2[6:9] = [0, 0, 0]
        subgoal2[6:11] = [0, 0, 0, 0.02, 0.02]
        return [subgoal1, subgoal2]


class CrowdSimAchiever(AbstractAchiever):
    def __init__(self, _range, n_obs, **params):
        super().__init__(_range, n_obs)  # range: dict{"dict", "angle"}
        self.subgoals = self.__generate_subgoals()

    def eval(self, state, current_state):
        """check whether a state is a subgoal 

        Args:
            state (ndarray): state from env.
            current_state (int): subgoal indicator.

        Returns:
            bool: is a state an subgoal
        """
        # state: JointState: self_state, human_states
        # TODO in the environment with more two humans
        if current_state >= len(self.subgoals):
            return False
        subgoal = self.subgoals[current_state]
        robot_state = state.self_state
        human_state = state.human_states[0]
        return self.check_subgoal(subgoal, robot_state, human_state)

    def check_subgoal(self, subgoal, robot_state, human_state):
        """check that the relative position vector betweem human and robot is
        in the range.

        Args:
            subgoal (dict): [description]
            robot_state (ndarray): [description]
            human_state (ndarray): [description]

        Returns:
            bool: [description]
        """
        # 人とロボットの相対位置ベクトルを算出し、そのベクトルがサブゴールに設定されている範囲内にあるかのチェック
        r_h_angle = self.__get_r_h_angle(robot_state, human_state)
        b_angle = self.__in_range(subgoal, r_h_angle, key="angle")
        return b_angle

    def __generate_subgoals(self):
        # 相対座標により指定
        # v_rとv_hの差分のcosが1；直角に交わるかつ、robotがcell_sizeよりhumanの後ろを通る。
        return [
            {
                "angle": 180,  # humanの速度ベクトルとhumanとrobotの相対座標[degree]
                "dist": 4  # 人のpositionを原点とした相対座標, CrowdSimから参照
            }
        ]

    def __calc_angle(self, vec1, vec2):
        """Return degree of angle between vec1 and vec2

        Args:
            vec1 ([list]): [base vector]
            vec2 ([list]): [description]

        Returns:
            [type]: [description]
        """
        cos = vec1[0]*vec2[0] + vec1[1]*vec2[1]
        sin = vec1[0]*vec2[1] - vec2[0]*vec1[1]
        atan2_rad = math.atan2(sin, cos)
        degree = math.degrees(atan2_rad)
        if degree < 0:
            degree = 360 + degree
        return degree

    def __calc_dist(self, vec1, vec2):
        """Calculate the distance between vectors.

        Args:
            vec1 (list): coordinate
            vec2 (list): coordinate

        Returns:
            int: distance
        """
        diff = np.array(vec1) - np.array(vec2)
        return np.linalg.norm(diff)

    def __in_range(self, basis, target, key=None):
        """Check whehter target is in the range.

        Args:
            basis (float): [description]
            target (float): [description]
            key (string, optional): [description]. Defaults to None.

        Returns:
            bool: [description]
        """
        if key is None:
            upper = basis + self._range
            lower = basis - self._range
        else:
            upper = basis[key] + self._range[key]
            lower = basis[key] - self._range[key]
        if key == "angle":
            upper = upper % 360
            lower = lower % 360
        return upper >= target and target >= lower

    def __get_r_h_angle(self, robot_state, human_state):
        r_h_rel_pos = [
            robot_state.px - human_state.px,
            robot_state.py - human_state.py
        ]
        human_vel = [human_state.vx, human_state.vy]
        if human_vel[0] == 0 and human_vel[1] == 0:
            # 速度が無い場合は0
            r_h_angle = +0
        else:
            r_h_angle = self.__calc_angle(human_vel, r_h_rel_pos)
        return r_h_angle
