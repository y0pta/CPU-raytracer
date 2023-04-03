"""
Thanks
https://raytracing.github.io/books/RayTracingInOneWeekend.html#surfacenormalsandmultipleobjects
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import random
import time, datetime
import concurrent.futures, multiprocessing
from ray import Ray
from camera import Camera
from hittable import *
from material import *


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


def calc_pixel(w: int, h: int, camera: Camera, obj_list: list, samples_per_pixel=15, max_reflections=15):
    pixel_color = np.array((0, 0, 0.0), dtype=np.double)

    du = [np.random.rand(1)[0] for i in range(samples_per_pixel)]
    dv = [np.random.rand(1)[0] for i in range(samples_per_pixel)]
    if samples_per_pixel > 4:
        du[:4] = [0, 0, 1, 1]
        dv[:4] = [0, 1, 0, 1]

    for s in range(0, samples_per_pixel):
        u = (w + du[s]) * camera.viewport_w / W - camera.viewport_w / 2
        v = (h + dv[s]) * camera.viewport_h / H - camera.viewport_h / 2

        r = camera.getRay(u, v)
        pixel_color += ray_color(r, obj_list, max_reflections)

        print(f"{w} {h} processed", end='\r')
    color = np.clip(np.sqrt(pixel_color / samples_per_pixel), 0, 1.0)

    return w, h, color


def scene1():
    # objects and materials
    ground = Sphere(np.array((0, -100.5, -1.0)), 100, Lambertian(np.array((0.8, 0.8, 0.0))))
    center_ball = Sphere(np.array((0, 0.0, -1.0)), 0.5, Lambertian(np.array((0.7, 0.3, 0.3))))
    left_ball = Sphere(np.array((-1.0, 0.0, -1.0)), 0.5, Dielectric(1.5))
    right_ball = Sphere(np.array((1.0, 0.0, -1.0)), 0.5, Metal(np.array((0.8, 0.6, 0.2))))
    # camera
    global look_at, look_from
    look_from = np.array((3, 3, 2), dtype=np.double)
    look_at = np.array((0, 0, -1), dtype=np.double)
    return [ground, center_ball, left_ball, right_ball]


def scene2():
    scene = []
    # objects and materials
    ground = Sphere(np.array((0, -1000, -1.0)), 1000, Lambertian(np.array((0.8, 0.4, 0.5))))
    scene.append(ground)

    a = 6
    b = 6
    for i in range(a):
        for j in range(b):
            x = - i * 2
            y = - j * 2
            r = 0.50 + np.random.rand(1)[0] / 4.0
            center = np.array((x, r, y))

            material = np.random.rand(1)[0]
            if material < 0.6:
                # diffuse
                albedo = np.random.rand(3)
                sphere_material = Lambertian(albedo)
                scene.append(Sphere(center, r, sphere_material))
            elif material < 0.85:
                # metal
                albedo = 0.5 + np.random.random(3) / 2.0
                fuzz = np.random.random(3) / 2.0
                sphere_material = Metal(albedo, fuzz)
                scene.append(Sphere(center, r, sphere_material))
            else:
                # glass
                sphere_material = Dielectric(1.5)
                scene.append(Sphere(center, r, sphere_material))
    # camera
    global look_at, look_from
    look_from = np.array((a / 2, 1.5 * a, b), dtype=np.double)
    look_at = np.array((-a/2, 0, -b), dtype=np.double)
    return scene


def gen_random_scene():
    world = []
    ground_material = Lambertian(np.array((0.5, 0.5, 0.5)))
    world.append(Sphere(np.array((0, -1000, -1.0)), 1000, ground_material))

    num_balls = 8
    for a in range(int(-num_balls/4), int(num_balls/4)):
        for b in range(int(-num_balls/4), int(num_balls/4)):
            center = np.array((a + 0.9 * np.random.rand(1)[0],
                              0.2,
                              b + 0.9 * np.random.rand(1)[0]))

            choose_mat = np.random.rand(1)[0]
            if np.linalg.norm(center - np.array((4, 0.2, 0))) > 0.9:
                if choose_mat < 0.8:
                    # diffuse
                    albedo = np.random.rand(3)
                    sphere_material = Lambertian(albedo)
                    world.append(Sphere(center, 0.2, sphere_material))
                elif choose_mat < 0.95:
                    # metal
                    albedo = 0.5 + np.random.random(3) / 2.0
                    fuzz = np.random.random(3) / 2.0
                    sphere_material = Metal(albedo, fuzz)
                    world.append(Sphere(center, 0.2, sphere_material))
                else:
                    # glass
                    sphere_material = Dielectric(1.5)
                    world.append(Sphere(center, 0.2, sphere_material))

    material1 = Dielectric(1.5)
    world.append(Sphere(np.array((0, 1, 0)), 1.0, material1))

    material2 = Lambertian(np.array((0.4, 0.2, 0.1)))
    world.append(Sphere(np.array((-4, 1, 0)), 1.0, material2))

    material3 = Metal(np.array((0.7, 0.6, 0.5)), 0.0)
    world.append(Sphere(np.array((4, 1, 0)), 1.0, material3))

    global look_at, look_from
    look_from = np.array((10, 2, 3), dtype=np.double)  # np.array((3, 3, 2), dtype=np.double)
    look_at = np.array((0.0, 0, 0), dtype=np.double)
    return world


fig, ax = plt.subplots()
# canvas set up
W = 640
H = 480
img = np.random.random((W, H, 3))
img = img.astype('double')

# sampling for antialiasing (will intersect each pixel multiple times in different areas inside the pixel)
SAMPLES_PER_PIXEL = 10
# camera
look_from = np.array((0, 0, 0), dtype=np.double)
look_at = np.array((0.0, 0, 0), dtype=np.double)
SCENE = scene2()
aperture = 1.0

focus_dist = np.linalg.norm(look_from - look_at)
cam = Camera(30, float(W) / H, aperture, 1, look_from, look_at)

# number of ray reflections
MAX_RAY_REFLECTIONS = 10


def draw(xycolor):
    new_pixels = len(xycolor)
    for i in range(new_pixels):
        x = xycolor[i][0]
        y = xycolor[i][1]
        color = xycolor[i][2]
        img[x, y] = color

    res = np.swapaxes(img, 1, 0)
    res = np.flip(res, 0)
    ax.clear()
    ax.imshow(res)
    plt.pause(0.001)


if __name__ == '__main__':
    start_time = time.perf_counter()
    print(img.shape)

    # fill the arguments for each thread
    all_w = [int(i / H) for i in range(0, H * W)]
    all_h = [i for i in range(0, H)] * W
    all_cam = [cam, ] * (W * H)
    scene = [SCENE, ] * (W * H)
    items = list(zip(all_w, all_h, all_cam, scene))

    num_processes = multiprocessing.cpu_count() - 1
    num_processed_pixels = 0
    batch = 15

    with multiprocessing.Pool(num_processes) as pool:
        while num_processed_pixels < W * H:
            start_idx = num_processed_pixels
            num_processed_pixels += num_processes * batch * num_processes
            end_idx = num_processed_pixels if num_processed_pixels < W * H else W * H

            results = pool.starmap(calc_pixel, items[start_idx: end_idx])
            draw(results)

    print("work done.")
    ex_time = time.perf_counter() - start_time
    print(f"Time passed: {ex_time}s")    # flip image y for show, because our coordinate system is from bottom left, instead of top left
    img = np.swapaxes(img, 1, 0)
    img = np.flip(img, 0)
    plt.ioff()
    plt.imsave(f"render-{int(ex_time)}s.png", img)
    print("Render saved.")
    plt.imshow(img)
    plt.show()