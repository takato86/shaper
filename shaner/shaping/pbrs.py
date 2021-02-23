

class PBRS:
    def __init__(self, gamma):
        self.gamma = gamma 
    
    def potential(self, obs):
        raise NotImplementedError
