import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import numpy as np

def visualize_3d(env):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for _ in range(200):
        ax.clear()
        # cube position
        ax.scatter(env.x, env.y, env.z, c="b", s=100, marker="o")
        # target
        tx, ty, tz = env.target_pos
        ax.scatter(tx, ty, tz, c="r", s=200, marker="x")
        ax.set_xlim(0, env.width)
        ax.set_ylim(0, env.height)
        ax.set_zlim(0, env.depth)
        ax.set_title("Hovercraft 3D")
        plt.draw()
        plt.pause(0.05)
        # avance un pas avec action random pour test
        env.step(np.random.randint(8))

    plt.show()
