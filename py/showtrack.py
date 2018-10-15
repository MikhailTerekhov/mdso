import numpy as np
import pandas as pd
import quaternion as quat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def extract_motions(fname):
    tbl = pd.read_csv(fname, sep='\s+', header=None,
                      index_col=0, dtype=np.float64)
    return list(map(
        lambda x: (np.array(x[-3:]),
                   quat.as_rotation_matrix(
                       np.quaternion(x[3], x[0], x[1], x[2]))),
        tbl.apply(list, axis=1).tolist()
    ))


def draw_motions(axes, motions, color):
    centers = np.array(
        list(map(lambda mot: -np.matmul(mot[1].T, mot[0]), motions)))
    dir_vects = np.array(list(map(lambda mot: 0.15 * mot[1][2, :], motions)))
    ax.quiver(centers[:, 0], centers[:, 1], centers[:, 2],
              dir_vects[:, 0], dir_vects[:, 1], dir_vects[:, 2], color=color)


actual = extract_motions('track.txt')
predicted = extract_motions('trackpred.txt')

fig = plt.figure()
ax = Axes3D(fig)

draw_motions(ax, actual, 'orange')
draw_motions(ax, predicted, 'blue')

plt.show()
