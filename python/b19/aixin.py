import random
from math import sin, cos, pi, log
from tkinter import *
import turtle

CANVAS_WIDTH = 640  # 画布的宽
CANVAS_HEIGHT = 480  # 画布的高
CANVAS_CENTER_X = CANVAS_WIDTH / 2  # 画布中心的X轴坐标
CANVAS_CENTER_Y = CANVAS_HEIGHT / 2  # 画布中心的Y轴坐标
IMAGE_ENLARGE = 11  # 放大比例
HEART_COLOR = "#fc8ca4"  # 心的颜色，这个是粉色


def heart_function(t, shrink_ratio: float = IMAGE_ENLARGE):
    """
    “爱心函数生成器”
    :param shrink_ratio: 放大比例
    :param t: 参数
    :return: 坐标
    """
    # 基础函数
    x = 16 * (sin(t) ** 3)
    y = -(13 * cos(t) - 5 * cos(2 * t) - 2 * cos(3 * t) - cos(4 * t))

    # 放大
    x *= shrink_ratio
    y *= shrink_ratio

    # 移到画布中央
    x += CANVAS_CENTER_X
    y += CANVAS_CENTER_Y

    return int(x), int(y)


def scatter_inside(x, y, beta=0.15):
    """
    随机内部扩散
    :param x: 原x
    :param y: 原y
    :param beta: 强度
    :return: 新坐标
    """
    ratio_x = - beta * log(random.random())
    ratio_y = - beta * log(random.random())

    dx = ratio_x * (x - CANVAS_CENTER_X)
    dy = ratio_y * (y - CANVAS_CENTER_Y)

    return x - dx, y - dy


def shrink(x, y, ratio):
    """
    抖动
    :param x: 原x
    :param y: 原y
    :param ratio: 比例
    :return: 新坐标
    """
    force = -1 / (((x - CANVAS_CENTER_X) ** 2 + (y - CANVAS_CENTER_Y) ** 2) **
                  0.6)  # 这个参数...
    dx = ratio * force * (x - CANVAS_CENTER_X)
    dy = ratio * force * (y - CANVAS_CENTER_Y)
    return x - dx, y - dy


def curve(p):
    """
    自定义曲线函数，调整跳动周期
    :param p: 参数
    :return: 正弦
    """
    # 可以尝试换其他的动态函数，达到更有力量的效果（贝塞尔？）
    return 2 * (2 * sin(4 * p)) / (2 * pi)


class Heart:
    """
    爱心类
    """

    def __init__(self, generate_frame=20):
        self._points = set()  # 原始爱心坐标集合
        self._edge_diffusion_points = set()  # 边缘扩散效果点坐标集合
        self._center_diffusion_points = set()  # 中心扩散效果点坐标集合
        self.all_points = {}  # 每帧动态点坐标
        self.build(2000)

        self.random_halo = 1000

        self.generate_frame = generate_frame
        for frame in range(generate_frame):
            self.calc(frame)

    def build(self, number):
        # 爱心
        for _ in range(number):
            t = random.uniform(0, 2 * pi)  # 随机不到的地方造成爱心有缺口
            x, y = heart_function(t)
            self._points.add((x, y))

        # 爱心内扩散
        for _x, _y in list(self._points):
            for _ in range(3):
                x, y = scatter_inside(_x, _y, 0.05)
                self._edge_diffusion_points.add((x, y))

        # 爱心内再次扩散
        point_list = list(self._points)
        for _ in range(4000):
            x, y = random.choice(point_list)
            x, y = scatter_inside(x, y, 0.17)
            self._center_diffusion_points.add((x, y))

    @staticmethod
    def calc_position(x, y, ratio):
        # 调整缩放比例
        force = 1 / (((x - CANVAS_CENTER_X) ** 2 +
                     (y - CANVAS_CENTER_Y) ** 2) ** 0.520)  # 魔法参数

        dx = ratio * force * (x - CANVAS_CENTER_X) + random.randint(-1, 1)
        dy = ratio * force * (y - CANVAS_CENTER_Y) + random.randint(-1, 1)

        return x - dx, y - dy

    def calc(self, generate_frame):
        ratio = 10 * curve(generate_frame / 10 * pi)  # 圆滑的周期的缩放比例

        halo_radius = int(4 + 6 * (1 + curve(generate_frame / 10 * pi)))
        halo_number = int(
            3000 + 4000 * abs(curve(generate_frame / 10 * pi) ** 2))

        all_points = []

        # 光环
        heart_halo_point = set()  # 光环的点坐标集合
        for _ in range(halo_number):
            t = random.uniform(0, 2 * pi)  # 随机不到的地方造成爱心有缺口
            x, y = heart_function(t, shrink_ratio=11.6)  # 魔法参数
            x, y = shrink(x, y, halo_radius)
            if (x, y) not in heart_halo_point:
                # 处理新的点
                heart_halo_point.add((x, y))
                x += random.randint(-14, 14)
                y += random.randint(-14, 14)
                size = random.choice((1, 2, 2))
                all_points.append((x, y, size))

        # 轮廓
        for x, y in self._points:
            x, y = self.calc_position(x, y, ratio)
            size = random.randint(1, 3)
            all_points.append((x, y, size))

        # 内容
        for x, y in self._edge_diffusion_points:
            x, y = self.calc_position(x, y, ratio)
            size = random.randint(1, 2)
            all_points.append((x, y, size))

        for x, y in self._center_diffusion_points:
            x, y = self.calc_position(x, y, ratio)
            size = random.randint(1, 2)
            all_points.append((x, y, size))

        self.all_points[generate_frame] = all_points

    def render(self, render_canvas, render_frame):
        for x, y, size in self.all_points[render_frame % self.generate_frame]:
            render_canvas.create_rectangle(
                x, y, x + size, y + size, width=0, fill=HEART_COLOR)


class Firework:
    def __init__(self):
        self.particles = []
        self.exploded = False
        self.start_x = random.randint(int(CANVAS_CENTER_X - CANVAS_WIDTH / 4),
                                      int(CANVAS_CENTER_X + CANVAS_WIDTH / 4))
        self.start_y = CANVAS_CENTER_Y  # 起始位置
        self.end_y = CANVAS_CENTER_Y + CANVAS_HEIGHT / 3  # 烟花上升的目标位置
        self.path = [(self.start_x, self.start_y)]

    def update(self):
        if not self.exploded:
            # 确保烟花向上移动
            new_y = self.start_y - 10  # 向上移动
            self.path.append((self.start_x, new_y))
            self.start_y = new_y
            if self.start_y <= self.end_y:
                self.explode()
        else:
            particles_to_remove = []
            for particle in self.particles:
                if particle.move():
                    particles_to_remove.append(particle)
            for particle in particles_to_remove:
                self.particles.remove(particle)
            if len(self.particles) == 0:
                fireworks.remove(self)

    def explode(self):
        self.exploded = True
        num_particles = 30
        for _ in range(num_particles):
            angle = random.uniform(0, 2 * pi)
            speed = random.uniform(10, 20)
            self.particles.append(Particle(self.start_x, self.start_y, angle, speed))

class Particle:
    def __init__(self, x, y, angle, speed):
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = speed
        self.life_span = 30  # 粒子的生命值

    def move(self):
        self.x += cos(self.angle) * self.speed
        self.y += sin(self.angle) * self.speed
        self.speed -= 0.5
        self.life_span -= 1
        if self.life_span <= 0:
            return True
        return False


def draw(main: Tk, render_canvas: Canvas, render_heart: Heart, render_frame=0):
    render_canvas.delete('all')
    render_heart.render(render_canvas, render_frame)

    if render_frame % 5 == 0:  # 每50帧添加一个新烟花
        fireworks.append(Firework())

    for firework in fireworks[:]:
        firework.update()
        for i in range(len(firework.path) - 1):
            # 绘制烟花轨迹为黄色
            render_canvas.create_line(firework.path[i][0], firework.path[i][1],
                                      firework.path[i + 1][0], firework.path[i + 1][1],
                                      fill="yellow")
        for particle in firework.particles:
            # 绘制粒子为黄色
            render_canvas.create_oval(particle.x - 2, particle.y - 2,
                                      particle.x + 2, particle.y + 2,
                                      fill="yellow")

    main.after(160, draw, main, render_canvas, render_heart, render_frame + 1)
    

if __name__ == '__main__':
    root = Tk()
    canvas = Canvas(root, bg='black', height=CANVAS_HEIGHT, width=CANVAS_WIDTH)
    canvas.pack()
    heart = Heart()
    fireworks = []  # 初始化烟花列表
    draw(root, canvas, heart)
    root.mainloop()
