from abc import ABCMeta


class AbstractAchiever(metaclass=ABCMeta):
    def eval(self, obs, subgoal_idx):
        pass
