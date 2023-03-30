import numpy as np


class Ray:
    def __init__(self, origin3D : np.array, direction3D : np.array):
        self.origin = origin3D
        self.direction = direction3D / np.linalg.norm(direction3D)

    def at(self, t):
        return self.origin + t * self.direction

