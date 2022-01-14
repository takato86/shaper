

class TableValue:
    def __init__(self, env, n_states, values=None):
        self.env = env
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
            return values

    def update(self, state, v):
        self.value[state] = v

    def get_min_value(self):
        return min(self.value.values())
