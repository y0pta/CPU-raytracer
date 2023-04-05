import math

import numpy as np
from .ray import Ray


def random_on_disk():
    """
    Generates random direction inside unit disk
    :return:
    """
    while True:
        res = np.random.rand(2)
        if np.linalg.norm(res) >= 1:
            continue
        else:
            return res


class Camera:
    """
    Camera model:
    -by default center located in (0, 0, 0);
    -image plane is on (0,0,-1);
    -image size viewport_w X viewport_h, where the lowest dimension is 1.0
    - depth of field
    """

    def __init__(self, fovw: float, aspect_ratio: float,
                 look_from=np.array((0.0, 0.0, 0.0)),
                 look_at=np.array((0.0, 0.0, -1.0)),
                 focus_dist=1.0,
                 aperture=0.0,
                 ):
        """
        :param fovw: horizontal field of view in DEGREES
        :param aspect_ratio: w/h
        :param focus_dist: being used for depth of field effect)
        :param aperture: size of hole in camera (influences for depth of field)
        :param look_from: origin of a camera
        :param look_at: target viewpoint
        """
        self.fovw = np.radians(fovw)
        half_w = np.tan(self.fovw / 2.0)
        # viewport width
        self.viewport_w = 2.0 * half_w
        # viewport height
        self.viewport_h = self.viewport_w / aspect_ratio
        # origin
        self.origin = look_from
        # view direction
        v = look_at - look_from
        self.view_dir = v / np.linalg.norm(v)
        # right vector
        up = np.array((0.0, 1.0, 0))
        r = np.cross(self.view_dir, up)
        self.right = r / np.linalg.norm(r)
        # up vector
        self.up = np.cross(self.right, self.view_dir)
        self.up = self.up / np.linalg.norm(self.up)
        # lens
        self.lens_radius = aperture / 2.0
        self.focus_dist = focus_dist

    def getRay(self, u, v) -> Ray:
        """
        Generates ray from u, v.
        :param u: u in [-viewport_w/2, viewport_w/2]
        :param v: v in [-viewport_h/2, viewport_h/2]
        :return: ray
        """
        assert -self.viewport_w / 2.0 <= u <= self.viewport_w / 2.0
        assert -self.viewport_h / 2.0 <= v <= self.viewport_h / 2.0

        rd = self.lens_radius * random_on_disk()
        offset = self.right * rd[0] + self.up * rd[1]

        return Ray(self.origin + offset,
                     u * self.right * self.focus_dist
                   + v * self.up * self.focus_dist
                   + self.view_dir * self.focus_dist
                   - offset)
