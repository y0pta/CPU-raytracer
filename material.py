from ray import Ray
import numpy as np
from hittable import HitInfo, Sphere


def reflect(v: np.array, n: np.array) -> np.array:
    """
     Reflects vector from surface with normal n
     :return: reflected vector
    """
    assert v.shape == n.shape
    return v - 2 * np.dot(v, n) * n


class Material:
    def __init__(self):
        pass

    def scatter(self, ray: Ray, hit_info: HitInfo) -> (Ray,np.array):
        """
        Calculates ray direction and color after hitting with object
        :return: scattered ray, color
        """
        pass


class Lambertian(Material):
    def __init__(self, color):
        # albedo is the amount of energy reflected by a surface is called. The darker surface has a lower albedo
        self.albedo = color

    def scatter(self, ray: Ray, hit_info: HitInfo) -> (Ray,np.array):
        scatter_direction = hit_info.normal + Sphere.random_direction()

        # if scatter direction near zero, amend it to normal direction (scatter light perpendicular)
        if np.linalg.norm(scatter_direction) < 0.01:
            scatter_direction = hit_info.normal

        scattered = Ray(hit_info.point, scatter_direction)

        return scattered, self.albedo


class Metal(Material):
    def __init__(self, color: np.array):
        self.albedo = color

    def scatter(self, ray: Ray, hit_info: HitInfo) -> (Ray, np.array):
        reflected = reflect(ray.direction, hit_info.normal)
        scattered = Ray(hit_info.point, reflected)

        return scattered, self.albedo

