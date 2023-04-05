from .scene_generator import Scene
from .ray import Ray
from .hittable import HitInfo, inf
from .camera import Camera
import numpy as np


class PixelInfo:
    def __init__(self, scene: Scene, size: tuple, pixel: tuple, samples_per_pixel=10, max_reflections=10):
        self.scene = scene
        self.W = size[0]
        self.H = size[1]
        self.x = pixel[0]
        self.y = pixel[1]
        self.samples_per_pixel = samples_per_pixel
        self.max_reflections = max_reflections
        self.color = None


def ray_color(ray: Ray, obj_list: list, depth: int):
    """
    Main function of color defining
    """
    if depth <= 0:
        return np.array((0, 0, 0))

    t_max = inf
    hit_info = HitInfo(-1)
    # find intersection with all objects in scene
    for obj in obj_list:
        hit_info_temp = obj.hit(ray, t_min=0.0001, t_max=t_max)
        if hit_info_temp.valid:
            # limit t above, because we don't need to calculate far objects
            t_max = hit_info_temp.t
            hit_info = hit_info_temp

    if hit_info.valid:
        scattered, attenuation = hit_info.material.scatter(ray, hit_info)
        color = attenuation * ray_color(scattered, obj_list, depth - 1)
        return color

    # intersection is not found
    t = 0.5 * (ray.direction[1] + 1)
    # blend color for sky gradient
    return (1.0 - t) * np.array((1.0, 1.0, 1.0)) + t * np.array((0.5, 0.7, 1))


def calc_pixel(info: PixelInfo):
    camera = info.scene.cam
    objects = info.scene.objects
    spp = info.samples_per_pixel

    pixel_color = np.array((0, 0, 0.0), dtype=np.double)

    du = [np.random.rand(1)[0] for i in range(spp)]
    dv = [np.random.rand(1)[0] for i in range(spp)]
    if spp > 4:
        du[:4] = [0, 0, 1, 1]
        dv[:4] = [0, 1, 0, 1]

    for s in range(0, spp):
        u = (info.x + du[s]) * camera.viewport_w / info.W - camera.viewport_w / 2
        v = (info.y + dv[s]) * camera.viewport_h / info.H - camera.viewport_h / 2

        r = camera.getRay(u, v)
        pixel_color += ray_color(r, objects, info.max_reflections)

        print(f"{info.x} {info.y} processed", end='\r')

    info.color = np.clip(np.sqrt(pixel_color / spp), 0, 1.0)
    return info.color