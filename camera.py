import math

import numpy as np
from ray import Ray


class Camera:
    """
    Camera model:
    -by default center located in (0, 0, 0);
    -image plane is on (0,0,-1);
    -image size viewport_w X viewport_h, where the lowest dimension is 1.0
    """

    def __init__(self, fovw: float, aspect_ratio: float):
        """
        :param fovy: horizontal field of view in DEGREES
        :param aspect_ratio: w/h
        """
        self.fovw = math.radians(fovw)
        half_w = math.tan(self.fovw/2.0)
        self.viewport_w = 2.0 * half_w
        self.viewport_h = self.viewport_w / aspect_ratio

        self.focal_length = 1

        self.origin = np.array((0.0, 0.0, 0.0))

        self.view_dir = np.array((0.0, 0.0, -1.0))

        self.horizontal = np.array((self.viewport_w, 0, 0))
        self.vertical = np.array((self.viewport_h, 0, 0))
        self.bl = self.origin - self.horizontal/2.0 - self.vertical/2.0 - np.array((0, 0, self.focal_length))

    def getRay(self, u, v) -> Ray:
        return Ray(self.origin, np.array((u, v, -self.focal_length)))
