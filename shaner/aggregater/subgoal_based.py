from shaner.aggregater.core import AbstractAggregater
from shaner.aggregater.entity.achiever import \
    FetchPickAndPlaceAchiever, \
    PinballAchiever, \
    FourroomsAchiever, \
    CrowdSimAchiever


id2achiever = {
    "SingleFetchPickAndPlace-v0": FetchPickAndPlaceAchiever,
    "FetchPickAndPlace-v1": FetchPickAndPlaceAchiever,
    "PinBall-v0": PinballAchiever,
    "Pinball-Subgoal-v0": PinballAchiever,
    "Fourrooms-v0": FourroomsAchiever,
    "ConstFourrooms-v0": FourroomsAchiever,
    "DiagonalFourrooms-v0": FourroomsAchiever,
    "DiagonalPartialFourrooms-v0": FourroomsAchiever,
    "CrowdSim-v0": CrowdSimAchiever
}


class DTA(AbstractAggregater):
    def __init__(self, env_id, n_obs, _range, **params):
        self.achiever = id2achiever[env_id](n_obs=n_obs, _range=_range,
                                            **params)
        self.current_state = 0
        self.n_states = len(self.achiever.subgoals) + 1

    def __call__(self, obs):
        if self.achiever.eval(obs, self.current_state):
            self.current_state += 1
            return self.current_state
        else:
            ret = self.current_state
            return ret

    def reset(self):
        self.current_state = 0

    def get_n_states(self):
        return self.n_states

