import matplotlib.pyplot as plt
import numpy as np
import time

import matplotlib
matplotlib.use('Qt5Agg')

plt.ion()  # turn on interactive mode

fig, ax = plt.subplots()
grid = np.zeros((10,10))
ax.imshow(grid, cmap='gray_r')
explored_scatter = ax.scatter([], [], c='blue', s=8)
path_line, = ax.plot([], [], 'y-', linewidth=2)

explored = []
path = []

for i in range(10):
    for j in range(10):
        explored.append((i, j))
        x, y = zip(*explored)
        explored_scatter.set_offsets(np.c_[y, x])
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.01)

plt.ioff()
plt.show()