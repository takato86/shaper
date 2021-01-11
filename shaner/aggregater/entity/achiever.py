import logging
import numpy as np

logger = logging.getLogger(__name__)


class FetchPickAndPlaceAchiever:
    def __init__(self, _range, n_obs):
        self._range = _range  # 0.01
        self.n_obs = n_obs
        self.subgoals = self.__generate_subgoals()  # n_obs=25
    
    def eval(self, obs, current_state):
        if len(self.subgoals) <= current_state:
            return False
        subgoal = np.array(self.subgoals[current_state])
        idxs = np.argwhere(subgoal == subgoal)
        b_lower = subgoal[idxs] - self._range <= obs[idxs]
        b_higher = obs[idxs] <= subgoal[idxs] + self._range
        res = all(b_lower & b_higher)
        if res:
            logger.info("Achieve the subgoal{}".format(current_state))
        return res
        # for li, mi in zip(subgoal[idxs], obs[idxs]):
        #     if li is None:
        #         continue
        #     if mi < li - self._range or li + self._range < mi:
        #         return False        
        # return True

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

