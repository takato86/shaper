
class TableValue:
    def __init__(self, env, transformer, **params):
        self.env = env
        self.transformer = transformer
        self.value = self.__init_value()

    def __call__(self, obs):
        ind = self.transformer(obs)
        return self.value[ind]

    def __init_value(self):
        return {
            i: 0
            for i in self.transformer.idxs
        }

    def update(self, obs, v):
        idx = self.transformer(obs)
        self.value[idx] = v
