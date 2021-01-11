from shaner.aggregater.core import AbstractTransformer
from shaner.aggregater.entity.achiever import FetchPickAndPlaceAchiever

id2achiever = {
    "FetchPickAndPlace-v1": FetchPickAndPlaceAchiever
}


class DTA(AbstractTransformer):
    def __init__(self, env_id, n_obs, _range):
        self.achiever = id2achiever[env_id](n_obs=n_obs, _range=_range)
        self.current_state = 0
        self.n_states = len(self.achiever.subgoals) + 1
    
    def __call__(self, obs, done):
        if self.achiever.eval(obs, self.current_state):
            self.current_state += 1
            return self.current_state
        else:
            ret = self.current_state
            if done:
                self.current_state = 0
            return ret

    def get_n_states(self):
        return self.n_states