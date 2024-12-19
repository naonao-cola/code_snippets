import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

# 设置图形大小和坐标轴范围
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.axis('off')

# 创建爱心形状的数据点
t = np.linspace(0, 2 * np.pi, 1000)
x = 16 * np.sin(t) ** 3
y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)

# 缩小爱心的初始尺寸
x = x / 20
y = y / 20

# 爱心线条
heart_line, = ax.plot(x, y, lw=2, color='red', alpha=0.5)

# 爱心填充
heart_fill = ax.fill(x, y, 'r', alpha=0.5)[0]

# 烟花效果
circles = []


def init():
    return heart_line, heart_fill,


def animate(i):
    global heart_fill  # 声明 heart_fill 为全局变量
    # 动态改变爱心的大小来模拟跳动效果
    scale = (np.sin(i / 10.0) + 1.5) / 2.5
    new_x = x * scale
    new_y = y * scale
    heart_line.set_data(new_x, new_y)

    # 移除旧的爱心填充
    heart_fill.remove()
    # 创建新的爱心填充
    heart_fill = ax.fill(new_x, new_y, 'r', alpha=0.5)[0]

    # 添加或更新烟花效果
    if i % 5 == 0:  # 每5帧添加一个新的烟花
        circle = Circle((np.random.uniform(-1.4, 1.4), np.random.uniform(-1.4, 1.4)), 0.05,
                        color=np.random.rand(3, ), alpha=0.5)
        circles.append(circle)
        ax.add_patch(circle)
    for c in circles:
        c.set_radius(c.get_radius() * 1.1)  # 增加半径模拟烟花绽放
        if c.get_radius() > 0.2:  # 如果烟花太大，则移除
            c.remove()
            circles.remove(c)

    return [heart_line, heart_fill] + circles


# 使用 blit=False 可以解决某些情况下动画元素无法正确更新的问题
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=200, interval=50, blit=False)

plt.show()