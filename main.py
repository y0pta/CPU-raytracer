import matplotlib.pyplot as plt
import numpy as np
from numpy import random
import time, datetime
import concurrent.futures, multiprocessing
from ray import Ray
from camera import Camera
from hittable import *
from material import *


def create_image(w, h):
    """
    Create 3-channeled image w x h x 3 and fulfill it with random values
    """
    f = lambda x: random.randint(0, 255) / 255.0

    img = np.full((h, w, 3), 1.0, dtype=np.float32)
    f_vec = np.vectorize(f)

    return f_vec(img)


def ray_color(ray: Ray, obj_list: list, depth: int):
    """
    Main function of color defining
    """
    if depth <= 0:
        return np.array((0.1, 0.1, 0.1))

    t_max = inf
    hit_info = HitInfo(-1)
    # find intersection with all objects in scene
    for obj in obj_list:
        hit_info_temp = obj.hit(ray, t_min=0.001, t_max=t_max)
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


def calc_pixel(w, h):
    pixel_color = np.array((0, 0, 0.0))
    for s in range(0, samples_per_pixel):
        u = (2.0 * (w + s / samples_per_pixel) - W) / H
        v = (2.0 * (h + s / samples_per_pixel) - H) / H

        r = cam.getRay(u, v)
        pixel_color += ray_color(r, obj_list, 10)

        print(f"{w} {h} processed", end='\r')
    return np.clip(pixel_color / samples_per_pixel, 0, 1.0) ** 0.5


W = 640
H = 480
# initial setup. here make assumption that h < w
viewport_w = W / H
viewport_h = 1.0
# sampling for antialiasing (will intersect each pixel multiple times in different areas inside the pixel)
samples_per_pixel = 5
# objects and materials
ground      = Sphere(np.array((    0, -100.5, -1.0)), 100, Lambertian(np.array((0.8, 0.8, 0.0))))
center_ball = Sphere(np.array((    0,    0.0, -1.0)), 0.5, Lambertian(np.array((0.7, 0.3, 0.3))))
left_ball   = Sphere(np.array(( -1.0,    0.0, -1.0)), 0.5, Metal(np.array((0.8, 0.8, 0.8))))
right_ball  = Sphere(np.array((  1.0,    0.0, -1.0)), 0.5, Metal(np.array((0.8, 0.6, 0.2))))
obj_list = [ground, center_ball, left_ball, right_ball]
# camera
cam = Camera(viewport_w, viewport_h)
origin = np.array((0.0, 0.0, 0.0))
# number of ray reflections
max_depth = 10


if __name__ == '__main__':
    start_time = time.perf_counter()
    img = np.random.random((W, H, 3))
    print(img.shape)

    # fill the arguments for each thread
    all_w = [int(i / H) for i in range(0, H * W)]
    all_h = [i for i in range(0, H)] * W
    items = list(zip(all_w, all_h))

    with multiprocessing.Pool(4) as pool:
        results = pool.starmap(calc_pixel, items)
        for i, result in enumerate(results):
            img[all_w[i], all_h[i]] = result

    ex_time = time.perf_counter() - start_time
    print(f"Time passed: {ex_time}s")
    # flip image y for show, because our coordinate system is from bottom left, instead of top left
    img = np.swapaxes(img, 1, 0)
    img = np.flip(img, 0)
    plt.imshow(img)

    dt = datetime.datetime
    dt = dt.today()
    plt.imsave(f"render-{int(ex_time)}s.png", img)
    plt.show()


def single_thread_main():
    start_time = time.perf_counter()
    W = 640
    H = 480
    img = np.random.random((W, H, 3))
    print(img.shape)

    # initial setup. here make assumption that h < w
    viewport_w = W / H
    viewport_h = 1.0

    # camera
    origin = np.array((0.0, 0.0, 0.0))
    cam = Camera(viewport_w, viewport_h)

    # sampling for antialiasing (will intersect each pixel multiple times in different areas inside the pixel)
    samples_per_pixel = 10;

    # objects
    obj_list = [Sphere(np.array((0, 0.0, -1.0)), 0.5), Sphere(np.array((0, -100.5, -1)), 100)]

    fig, ax = plt.subplots()

    for i in range(0, W):
        for j in range(0, H):
            pixel_color = np.array((0, 0, 0.0))
            for s in range(0, samples_per_pixel):
                # our local coordinates with multi sampling

                u = (2.0 * (i + s / samples_per_pixel) - W) / H
                v = (2.0 * (j + s / samples_per_pixel) - H) / H

                #u = (2.0 * i - W) / H
                #v = (2.0 * j - H) / H

                r = cam.getRay(u, v)
                pixel_color += ray_color(r, obj_list)

            img[i, j] = pixel_color/samples_per_pixel

        print(f"W = {i} processed")
        ax.clear()
        ax.imshow(img)
        ax.set_title(f"row {i} processed")
        plt.pause(0.001)

    print(time.perf_counter() - start_time)
    # flip image y for show, because our coordinate system is from bottom left, instead of top left
    result = np.flip(img, 1)
    ax.imshow(result)
    plt.show()


def pre_emptive_threads():
    start_time = time.perf_counter()
    W = 640
    H = 480
    img = np.random.random((W, H, 3))
    print(img.shape)

    # initial setup. here make assumption that h < w
    viewport_w = W / H
    viewport_h = 1.0

    # camera
    origin = np.array((0.0, 0.0, 0.0))
    cam = Camera(viewport_w, viewport_h)

    # sampling for antialiasing (will intersect each pixel multiple times in different areas inside the pixel)
    samples_per_pixel = 10;

    # objects
    obj_list = [Sphere(np.array((0, 0.0, -1.0)), 0.5), Sphere(np.array((0, -100.5, -1)), 100)]

    fig, ax = plt.subplots()

    def calc_pixel(w, h):
        pixel_color = np.array((0, 0, 0.0))
        for s in range(0, samples_per_pixel):
            u = (2.0 * (w + s / samples_per_pixel) - W) / H
            v = (2.0 * (h + s / samples_per_pixel) - H) / H

            r = cam.getRay(u, v)
            pixel_color += ray_color(r, obj_list)

            print(f"{w} {h} processed", end='\r')
        return pixel_color / samples_per_pixel

    # fill the arguments for each thread
    all_w = [int(i / H) for i in range(0, H * W)]
    all_h = [i for i in range(0, H)] * W

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = executor.map(calc_pixel, all_w, all_h)
        for i, result in enumerate(results):
            img[all_w[i], all_h[i]] = result

    print(time.perf_counter() - start_time)
    # flip image y for show, because our coordinate system is from bottom left, instead of top left
    result = np.flip(img, 1)
    result = np.flip(img, 0)
    ax.imshow(result)
    plt.show()