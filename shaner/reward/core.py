

class HighReward:
    def __init__(self, gamma):
        self.value = 0
        self.gamma = gamma

    def __call__(self):
        return self.value

    def update(self, reward):
        self.value = self.gamma * self.value + reward

    def reset(self):
        self.value = 0
