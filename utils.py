import numpy as np

def snap_to_grid(x, resolution):
    return (resolution * round(x[0] / resolution), resolution * round(x[1] / resolution))