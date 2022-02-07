from abc import ABCMeta, abstractmethod


class AbstractShaping(metaclass=ABCMeta):
    @abstractmethod
    def shape(self, pre_obs, pre_action, reward, obs, done, info):
        pass
