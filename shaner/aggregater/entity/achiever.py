import logging
import numpy as np
import pandas as pd
import os
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
        self.subgoals = subgoals  # 2d-ndarray shape(#obs, #subgoals)

    def eval(self, obs, current_state):
        if len(self.subgoals) <= current_state:
            return False
        subgoal = np.array(self.subgoals[current_state])
        return all(obs == subgoal)


class PinballAchiever(AbstractAchiever):
    def __init__(self, _range, n_obs, subgoals, **params):
        super().__init__(_range, n_obs)
        self.subgoals = subgoals # 2d-ndarray shape(#obs, #subgoals)

    def eval(self, obs, current_state):
        if len(self.subgoals) <= current_state:
            return False
        subgoal = np.array(self.subgoals[current_state])
        idxs = np.argwhere(subgoal == subgoal)  # np.nanでない要素を取り出し
        b_in = l2_norm_dist(subgoal[idxs].reshape(-1), obs[idxs].reshape(-1)) <= self._range
        res = np.all(b_in)
        if res:
            logger.debug("Achieve the subgoal{}".format(current_state))
        return res

    def __generate_subgoals(self):
        df = pd.read_csv(self.subgoal_path)
        subgoals = df.values
        return subgoals


class FetchPickAndPlaceAchiever(AbstractAchiever):
    def __init__(self, _range, n_obs, **params):
        super().__init__(_range, n_obs)
        self.subgoals = self.__generate_subgoals()  # n_obs=25

    def eval(self, obs, current_state):
        if len(self.subgoals) <= current_state:
            return False
        subgoal = np.array(self.subgoals[current_state])
        idxs = np.argwhere(subgoal == subgoal) # np.nanでない要素を取り出し
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
        # state: JointState: self_state, human_states
        # TODO in the environment with more two humans
        if current_state >= len(self.subgoals):
            return False
        subgoal = self.subgoals[current_state]
        robot_state = state.self_state
        human_state = state.human_states[0]
        return self.check_subgoal_v6(subgoal, robot_state, human_state)

    def check_subgoal_v0(self, subgoal, robot_state, human_state):
        robot_coord = [robot_state.px, robot_state.py]
        human_coord = [human_state.px, human_state.py]
        r_h_dist = self.__calc_dist(robot_coord, human_coord)
        b_dist = self.__in_range(subgoal, r_h_dist, key="dist")
        return b_dist
    
    def check_subgoal_v1(self, subgoal, robot_state, human_state):
        r_h_anble = self.__get_r_h_angle(robot_state, human_state)
        b_angle = self.__in_range(subgoal, r_h_angle, key="angle")
        return b_angle

    def check_subgoal_v2(self, subgoal, robot_state, human_state):
        r_h_angle = self.__get_r_h_angle(robot_state, human_state)
        b_angle = self.__in_range(subgoal, r_h_angle, key="angle")
        return b_angle

    def check_subgoal_v4(self, subgoal, robot_state, human_state):
        r_h_angle = self.__get_r_h_angle(robot_state, human_state)
        b_angle = self.__in_range(subgoal, r_h_angle, key="angle")
        robot_coord = [robot_state.px, robot_state.py]
        human_coord = [human_state.px, human_state.py]
        r_h_dist = self.__calc_dist(robot_coord, human_coord)
        b_dist = self.__in_range(subgoal, r_h_dist, key="dist")
        return b_angle and b_dist

    def check_subgoal_v5(self, subgoal, robot_state, human_state):
        subgoal1 = {'angle': 0}
        subgoal2 = {'angle': 180}
        r_h_angle = self.__get_r_h_angle(robot_state, human_state)
        b_angle_1 = self.__in_range(subgoal1, r_h_angle, key="angle")
        b_angle_2 = self.__in_range(subgoal2, r_h_angle, key="angle")
        robot_coord = [robot_state.px, robot_state.py]
        human_coord = [human_state.px, human_state.py]
        r_h_dist = self.__calc_dist(robot_coord, human_coord)
        b_dist = self.__in_range(subgoal, r_h_dist, key="dist")
        return (b_angle_1 or b_angle_2) and b_dist

    def check_subgoal_v6(self, subgoal, robot_state, human_state):
        r_h_angle = self.__get_r_h_angle(robot_state, human_state)
        b_angle = self.__in_range(subgoal, r_h_angle, key="angle")
        robot_coord = [robot_state.px, robot_state.py]
        human_coord = [human_state.px, human_state.py]
        r_h_dist = self.__calc_dist(robot_coord, human_coord)
        b_dist = r_h_dist > 1.5  # 近づきすぎない
        return b_angle and b_dist

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
        diff = np.array(vec1) - np.array(vec2)
        return np.linalg.norm(diff)

    def __in_range(self, basis, target, key=None):
        if key is None:
            upper = basis + self._range
            lower = basis - self._range
        else:
            upper = basis[key] + self._range[key]
            lower = basis[key] - self._range[key]
        if key == "angle":
            upper = upper % 360
            lower = lower % 360
        return  upper >= target and target >= lower

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
