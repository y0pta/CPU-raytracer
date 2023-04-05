"""
Thanks
https://raytracing.github.io/books/RayTracingInOneWeekend.html#surfacenormalsandmultipleobjects
"""

import matplotlib.pyplot as plt
from numpy import random
import time
import multiprocessing
from source.camera import Camera
from source.hittable import *
from source.material import *
from source.scene_generator import Scene
from source.calc_pixel import *


def draw(ax, pinfo_list: list):
    for i in range(len(pinfo_list)):
        x = pinfo_list[i].x
        y = pinfo_list[i].y
        color = pinfo_list[i].color
        img[x, y] = color

    res = np.swapaxes(img, 1, 0)
    res = np.flip(res, 0)
    ax.clear()
    ax.imshow(res)
    plt.pause(0.001)


if __name__ == '__main__':
    # canvas set up
    W = 1024
    H = 600
    img = np.random.random((W, H, 3))
    img = img.astype('double')
    print(f"Image shape: {img.shape}")
    # scene set up
    scene = Scene(2, W, H)
    # rendering set up
    max_reflections = 20
    samples_per_pixel = 40
    num_processes = multiprocessing.cpu_count() - 1
    batch = 25
    # for plotting
    fig, ax = plt.subplots()

    start_time = time.perf_counter()
    # fill the arguments for each process
    pixels = []
    for i in range(W):
        for j in range(H):
            pixels.append(PixelInfo(scene, (W, H), (i, j), samples_per_pixel, max_reflections))

    pixels_done = 0
    with multiprocessing.Pool(num_processes) as pool:
        while pixels_done < W * H:
            start_idx = pixels_done
            pixels_done += num_processes * batch
            end_idx = pixels_done if pixels_done < W * H else W * H

            res = pool.map(calc_pixel, list(pixels[start_idx: end_idx]))
            for i in range(len(res)):
                pixels[start_idx + i].color = res[i]

            draw(ax, pixels[start_idx: end_idx])

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