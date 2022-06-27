
class HighReward:
    def __init__(self, gamma: float):
        if gamma >= 1:
            raise Exception("Expected gamma is <1, the input is >=1.")
        self.value = 0.0
        self.gamma = gamma

    def __call__(self) -> float:
        return self.value

    def update(self, reward: float, t: int) -> None:
        self.value += self.gamma**t * reward

    def reset(self) -> None:
        self.value = 0
