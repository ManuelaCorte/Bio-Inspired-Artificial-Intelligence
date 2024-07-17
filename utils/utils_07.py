import matplotlib.pyplot as plt
from matplotlib import collections as mc
from numpy.typing import NDArray
import numpy as np


def readFileAsMatrix(file: str) -> list[list[float]]:
    with open(file) as f:
        lines = f.read().splitlines()
        matrix: list[list[float]] = []
        for line in lines:
            row: list[float] = []
            for value in line.split():
                row.append(float(value))
            matrix.append(row)
        return matrix


def readFileAsList(file: str) -> NDArray[np.uint32]:
    with open(file) as f:
        lines = f.read().splitlines()
        n = len(lines)
        array = np.empty(n, dtype=np.uint32)
        for i in range(n):
            array[i] = int(lines[i])
        return array


def plotSolution(
    points: list[tuple[float, float]],
    distances: list[list[float]],
    solution: list[int],
    title: str,
):
    fig = plt.figure(title)
    ax = fig.add_subplot(111)
    ax.scatter(*zip(*points))

    for i, p in enumerate(points):
        ax.annotate(str(i), p)

    # draw all possible path segments
    lines = []
    for i, p in enumerate(points):
        for j, _ in enumerate(points):
            if distances[i][j] > 0 and i > j:
                lines.append((points[i], points[j]))
    lc = mc.LineCollection(lines, linewidths=0.1)
    ax.add_collection(lc)

    # draw the solution
    lines = []
    for i in np.arange(len(solution) - 1):
        lines.append((points[solution[i]], points[solution[i + 1]]))
    lines.append((points[solution[0]], points[solution[-1]]))
    lc = mc.LineCollection(lines, linewidths=1, color="r")
    ax.add_collection(lc)

    # ax.set_title(title)
    ax.autoscale()
    ax.margins(0.1)
