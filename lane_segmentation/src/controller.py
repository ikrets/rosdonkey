import numpy as np

class SmoothingController:
    def __init__(self, K, gain):
        self.gain = gain
        self.K = K / np.sum(K)
        self.reset()

    def reset(self):
        self.errors = np.zeros_like(self.K)

    def control(self, target, current):
        self.errors[1:] = self.errors[:-1]
        self.errors[0] = target - current

        return np.sum(self.gain * self.K * self.errors)

