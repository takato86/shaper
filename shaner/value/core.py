from shaner.utils import get_box

class TableValue:
    def __init__(self, env, **params):
        self.env = env
        self.n_states = get_box(env.observation_space).shape[0]
        self.value = self.__init_value()
        
    def __call__(self, state):
        return self.value[state]

    def __init_value(self):
        return {
            i: 0
            for i in range(self.n_states)
        }

    def update(self, state, v):
        self.value[state] = v
