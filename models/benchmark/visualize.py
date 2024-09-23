import sys
sys.path.append('/home/hienvq/Desktop/Lab/NNA_MPOC/')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

from models.benchmark.basic import *
from models.benchmark.functions import Functions

if __name__ == '__main__':
    ax = plt.axes(projection='3d')
    X = np.arange(-100, 100 + sys.float_info.min, 1)
    Y = np.arange(-100, 100 + sys.float_info.min, 1)
    X, Y = np.meshgrid(X, Y)
    XY = np.stack((X, Y), axis=-1)
    function = Functions('uni2', 2).value
    # function = schaffer_f6
    Z = np.apply_along_axis(function, 2, XY)
    # print(Z)
    my_col = cm.jet((Z - np.amin(Z))/(np.amax(Z)-np.amin(Z)))

    surf = ax.plot_surface(X, Y, Z, facecolors = my_col,
                           linewidth=0, antialiased=False)

    plt.show()
