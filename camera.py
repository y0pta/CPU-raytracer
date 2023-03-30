import numpy as np
from ray import Ray


class Camera:
    """
    Camera model:
    -by default center located in (0, 0, 0);
    -image plane is on (0,0,-focal.length);
    -image size viewport_w X viewport_h, where the lowest dimension is 1.0
    """

    def __init__(self, viewport_w, viewport_h, focal_length=1.0, origin=np.array((0.0,0.0,0.0))):
        self.w = viewport_w
        self.h = viewport_h
        self.focal_length = focal_length
        self.origin = origin
        self.horizontal = np.array((viewport_w, 0, 0))
        self.vertical = np.array((viewport_h, 0, 0))
        self.bl = origin - self.horizontal/2.0 - self.vertical/2.0 - np.array((0, 0, self.focal_length))

    def getRay(self, u, v) -> Ray:
        return Ray(self.origin, np.array((u, v, -self.focal_length)))
