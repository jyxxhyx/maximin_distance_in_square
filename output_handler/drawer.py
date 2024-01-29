import functools
import itertools

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon


def draw_single_plot(file_name, coords, min_dist, epsilon: float = 1e-5):
    fig, ax = plt.subplots()
    arcs = list()
    grid_x = [0, 1, 1, 0, 0]
    grid_y = [0, 0, 1, 1, 0]
    plt.plot(grid_x, grid_y, marker='', color='k', linewidth=1, alpha=0.3)

    for point1, point2 in itertools.combinations(coords, 2):
        man_distance = abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])
        if abs(man_distance - min_dist) < epsilon:
            arcs.append((point1, point2))

    for (point1, point2) in arcs:
        x = [point1[0], point2[0]]
        y = [point1[1], point2[1]]
        plt.plot(x, y, marker='', color='k', linewidth=3, alpha=0.5)

    x = [coord[0] for coord in coords]
    y = [coord[1] for coord in coords]
    plt.scatter(x, y, marker='o', color='r', s=121)

    for coord_x, coord_y in zip(x, y):
        plt.text(coord_x + 0.02, coord_y + 0.02, f'{coord_x:.4f},{coord_y:.4f}', fontsize=12)

    ax.set_xlim([-0.2, 1.2])
    ax.set_ylim([-0.2, 1.2])
    ax.axis('equal')
    ax.set_axis_off()
    ax.set_title(f'N = {len(coords)}, Min dist = {min_dist:.4f}',
                 fontsize=20)
    # plt.margins(0, 0)
    plt.savefig('{}.jpg'.format(file_name), bbox_inches='tight', pad_inches=0)
    plt.savefig('{}.pdf'.format(file_name), bbox_inches='tight', pad_inches=0)
    return
