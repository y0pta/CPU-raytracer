import numpy as np
from numpy import random
from ray import Ray
#from material import Material

inf = np.iinfo(np.int32).max
minus_inf = np.iinfo(np.int32).min


class Hittable:
    def hit(self, ray: Ray, t_min=minus_inf, t_max=inf):
        pass


class HitInfo:
    def __init__(self, t, point3D=None, normal3D=None, material=None, front_face=True):
        self.point = point3D
        self.normal = normal3D
        self.material = material
        self.front_face = front_face
        self.t = t

        if t == -1:
            self.valid = False
        else:
            self.valid = True


class Sphere(Hittable):
    def __init__(self, origin: np.array, radius: int, material):
        self.origin = origin
        self.radius = radius
        self.material = material

    def hit(self, ray: Ray, t_min=minus_inf, t_max=inf) -> HitInfo:
        """
             Solving square equation we can understand, if ray intersects the ray or not
             Flip normal if needed.
         """
        oc = ray.origin - self.origin
        a = np.linalg.norm(ray.direction) ** 2
        b = 2.0 * np.dot(ray.direction, oc)
        c = np.linalg.norm(oc) ** 2 - self.radius ** 2

        # find discriminant
        D = b ** 2 - 4 * a * c
        if D < 0:
            return HitInfo(-1)

        # D > 0, find roots
        root = (-b - np.sqrt(D)) / (2.0 * a)
        if root < t_min or root > t_max:
            root = (-b + np.sqrt(D)) / (2.0 * a)
            if root < t_min or root > t_max:
                return HitInfo(-1)

        # find normal points outward
        normal = ray.at(root) - self.origin
        normal = normal / np.linalg.norm(normal)

        # correct normal: if ray intersects sphere from inside out - flip normal
        front_face = np.dot(ray.direction, normal) <= 0.0
        normal = normal if front_face else -normal

        return HitInfo(root, ray.at(root), normal, self.material, front_face)

    @staticmethod
    def random_direction():
        """
        Generates random direction on sphere. All values are in [-1, 1]
        """
        r = np.random.random(3) * 2 - 1
        r = r.astype('double') / np.linalg.norm(r)
        return r

    @staticmethod
    def random_hemisphere_dir(normal):
        """
        Generates random direction on hemisphere.
        """
        r = np.random.random(3) * 2 - 1
        r[1] = r[1] if r[1] > 0 else -r[1]
        r = r.astype('double') / np.linalg.norm(r)
        return r if np.dot(r, normal) > 0.0 else -r
