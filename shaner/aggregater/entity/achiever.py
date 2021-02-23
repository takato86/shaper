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
        # extract coordinates
        robot_coord = [robot_state.px, robot_state.py]
        human_coord = [human_state.px, human_state.py]
        # extract velocity
        robot_vel = [robot_state.vx, robot_state.vy]
        human_vel = [human_state.vx, human_state.vy]
        r_h_dist = self.__calc_dist(robot_coord, human_coord)
        r_h_angle = abs(self.__calc_angle(robot_vel, human_vel))
        b_dist = self.__in_range(subgoal, r_h_dist, key="dist")
        b_angle = self.__in_range(subgoal, r_h_angle, key="angle")
        return b_dist and b_angle

    def __generate_subgoals(self):
        # 相対座標により指定
        # v_rとv_hの差分のcosが1；直角に交わるかつ、robotがcell_sizeよりhumanの後ろを通る。
        return [
            {
                "angle": 90,  # 相対速度ベクトルの偏角[degree]
                "dist": 2  # 人のpositionを原点とした相対座標, CrowdSimから参照
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
        return math.degrees(atan2_rad) 

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
        return  upper >= target and target >= lower
