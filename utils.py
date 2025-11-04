import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

def snap_to_grid(x, resolution):
    return (resolution * round(x[0] / resolution), resolution * round(x[1] / resolution))

