from .hittable import Sphere
from .material import *
from .camera import Camera


class Scene:
    def __init__(self, scene_num, W, H):
        self.objects = []
        self.cam = None

        aperture = 0.1
        focus_dist = 10
        look_from = np.array((0, 0, 0), dtype=np.double)
        look_at = np.array((0, 0, -1), dtype=np.double)
        fov = 90

        if scene_num == 1:
            self.objects = scene1()
            look_from = np.array((3, 3, 2), dtype=np.double)
            look_at = np.array((0, 0, -1), dtype=np.double)
            fov = 30
            aperture = 0.2
            focus_dist = np.linalg.norm(look_from - look_at)

        elif scene_num == 2:
            self.objects = scene2()
            look_from = np.array((-2, 7, -10), dtype=np.double)
            look_at = np.array((-6, 0.5, -6), dtype=np.double)
            fov = 50
            focus_dist = np.linalg.norm(look_from - look_at)
            aperture = 0.2

        elif scene_num == 3:
            self.objects = scene3()
            look_from = np.array((13, 2, 3), dtype=np.double)
            look_at = np.array((0.0, 0, 0), dtype=np.double)
            fov = 30
            focus_dist = 10

        self.cam = Camera(fov, float(W) / H, look_from, look_at, focus_dist, aperture)

    def resetCamera(self, vfov, W, H, look_from, look_at, focus_dist, aperture):
        self.cam = Camera(vfov, W / H, look_from, look_at, focus_dist, aperture)

    def resetCamera(self, camera: Camera):
        self.cam = camera


def scene1():
    # objects and materials
    ground = Sphere(np.array((0, -100.5, -1.0)), 100, Lambertian(np.array((0.8, 0.8, 0.0))))
    center_ball = Sphere(np.array((0, 0.0, -1.0)), 0.5, Lambertian(np.array((0.7, 0.3, 0.3))))
    left_ball = Sphere(np.array((-1.0, 0.0, -1.0)), 0.5, Dielectric(1.5))
    right_ball = Sphere(np.array((1.0, 0.0, -1.0)), 0.5, Metal(np.array((0.8, 0.6, 0.2))))

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
            r = 0.5 + np.random.rand(1)[0] / 4.0
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

    return scene


def scene3():
    world = []
    ground_material = Lambertian(np.array((0.5, 0.5, 0.5)))
    world.append(Sphere(np.array((0, -1000, -1.0)), 1000, ground_material))

    for a in range(-11, 11):
        for b in range(-11, 11):
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

    return world
