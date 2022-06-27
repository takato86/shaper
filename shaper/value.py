import abc


class AbstractValue(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, state: any) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, state: any, reward: float) -> None:
        raise NotImplementedError


class TableValue(AbstractValue):
    def __init__(self, n_states, values=None):
        self.n_states = n_states
        self.value = self.__init_value(values=values)

    def __call__(self, state):
        return self.value[state]

    def __init_value(self, values):
        if values is None:
            return {
                i: 0
                for i in range(self.n_states)
            }
        else:
            if len(values) != self.n_states:
                raise Exception(
                    "Not match the size of values: {} with the number of states: {}".format(
                        len(values), self.n_states
                    )
                )
            values = {
                int(key): value for key, value in values.items()
            }
            return values

    def update(self, state, v):
        self.value[state] = v

    def get_min_value(self):
        return min(self.value.values())
