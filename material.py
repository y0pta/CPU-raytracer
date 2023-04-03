import math

from ray import Ray
import numpy as np
from hittable import HitInfo, Sphere


def reflect(v: np.array, n: np.array) -> np.array:
    """
     Reflects vector from surface with normal n
    :param v: vector
    :param n: surface normal (vector comes into)
    :return: reflected vector
    """
    assert v.shape == n.shape
    return v - 2 * np.dot(v, n) * n


def refract(v: np.array, n: np.array, refraction_ratio: float) -> np.array:
    """
     Refract vector on the surface with normal n according to Snell's law
     :param v: vector
     :param n: surface normal (vector comes into)
     :param  refraction_ratio: refractive index of environment vector comes from divided by refractive index of
     environment vector comes in
     :return: refracted vector
    """
    assert np.dot(v, n) < 1.0
    res_x = refraction_ratio * (v - n * np.dot(v, n))
    assert np.linalg.norm(res_x) < 1.0
    res_y = -(1.0 - np.linalg.norm(res_x)**2)**0.5 * n

    return res_x + res_y


def reflectance(cosine: float, refractive_idx: float) -> float:
    """
    Schlick Approximation for glass
    :param cosine:
    :param refractive_idx:
    :return:
    """
    r0 = (1.0 - refractive_idx) / (1.0 + refractive_idx)
    r0 = r0**2
    return r0 + (1.0 - r0) * (1 - cosine)**5


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

    def scatter(self, ray: Ray, hit_info: HitInfo) -> (Ray, np.array):
        scatter_direction = hit_info.normal + Sphere.random_hemisphere_dir(hit_info.normal)
        if np.linalg.norm(scatter_direction) < 0.001 * np.linalg.norm(hit_info.normal):
            scatter_direction = hit_info.normal
        scattered = Ray(hit_info.point, scatter_direction)
        return scattered, self.albedo


class Metal(Material):
    def __init__(self, color: np.array, fuzz: float = 0):
        self.albedo = color
        self.fuzz = np.clip(fuzz, 0, 1)

    def scatter(self, ray: Ray, hit_info: HitInfo) -> (Ray, np.array):
        reflected = reflect(ray.direction, hit_info.normal)
        scattered = Ray(hit_info.point, reflected + self.fuzz * Sphere.random_direction())

        return scattered, self.albedo


class Dielectric(Material):
    def __init__(self, refraction_idx):
        # for air=1.0, glass=1.3-1.7, diamond=2.4
        self.ir = refraction_idx

    def scatter(self, ray: Ray, hit_info: HitInfo) -> (Ray, np.array):
        attenuation = np.array((1.0, 1.0, 1.0))
        refraction_ratio = 1.0 / self.ir if hit_info.front_face else self.ir

        # check if refraction exists (if angle is close to normal sometimes sin could be > 1)
        cos_ = np.dot(ray.direction, hit_info.normal)
        sin_ = math.sqrt(1.0 - cos_**2)

        # define if refraction possible
        cannot_refract = refraction_ratio * sin_ > 1.0
        v = ray.direction
        if cannot_refract or reflectance(cos_, refraction_ratio) > np.random.random(1)[0]:
            v = reflect(ray.direction, hit_info.normal)
        else:
            v = refract(ray.direction, hit_info.normal, refraction_ratio)

        scattered = Ray(hit_info.point, v)
        return scattered, attenuation

