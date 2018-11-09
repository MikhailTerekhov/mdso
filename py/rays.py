import sys
from operator import add
import numpy as np
import pandas as pd
import quaternion as quat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

fig = plt.figure()
ax = Axes3D(fig)

def plot_corresp(ray1, ray2, t, R):
    global fig, ax

    ax.clear()
    lines = [None] * 2
    lines[0] = np.column_stack((np.array([0, 0, 0]), 3 * np.array(ray1)))
    start2 = np.matmul(R.T, -np.array(t))
    dir2 = np.matmul(R.T, np.array(ray2))
    #  print(dir2)
    lines[1] = np.column_stack((start2, start2 + dir2))

    #  print(lines[1].T.tolist())
    ax.plot(*lines[0].tolist(), color='green')
    ax.plot(*lines[1].tolist(), color='orange')
    ax.scatter(*lines[0][:,0])
    ax.scatter(*lines[1][:,0])

    plane1 = list(map(tuple, [lines[0][:,0], lines[1][:,0], lines[0][:,1]]))
    plane2 = list(map(tuple, [lines[1][:,0], lines[0][:,0], lines[1][:,1]]))
    polycol = Poly3DCollection([plane1, plane2])
    polycol.set_facecolor((0,0,1,0.4))
    #  print(plane1)
    ax.add_collection3d(polycol)
    
    fig.canvas.draw()
    fig.canvas.flush_events()
#  draw_motions(ax, actual, 'orange')
#  draw_motions(ax, predicted, 'blue')n.

#  verts = [[(0, 0, 0), (0, 0, 1), (0, 1, 0)], [(1, 0, 0), (1, 0, 1), (1, 1, 0)]]
#  col = Poly3DCollection(verts, linewidth=1, edgecolors='k')
#  col.set_facecolor((0,0,1,0.1))

#  ax.add_collection3d(col)

total = 1
cur_ind = 0

num = []
rays1 = []
rays2 = []
t = []
R = []

def redraw():
    global total, cur_ind, rays1, rays2, t, R

    plot_corresp(rays1[cur_ind], rays2[cur_ind], t[cur_ind], R[cur_ind])

def on_key_pressed(event):
    global cur_ind, total

    if (event.key == 'left'):
        print('left!')
        cur_ind = (cur_ind - 1) % total
        redraw()
    elif (event.key == 'right'):
        print('right')
        cur_ind = (cur_ind + 1) % total
        redraw()

cid = fig.canvas.mpl_connect('key_press_event', on_key_pressed)

def main(argv):
    if (len(argv) != 2):
        print("I need just one argument -- dso output directory")
        return 0

    out_dir = argv[1]
    if (out_dir[-1] == '/'):
        out_dir = out_dir[0:-1]
    f = open(out_dir + "/corresps.txt")
    
    global total, cur_ind, rays1, rays2, t, R

    lines = f.readlines()
    total = len(lines) // 5
    print(f"total correspondencies = {total}")
    num = [0] * total
    rays1 = [(0, 0, 0)] * total
    rays2 = [(0, 0, 0)] * total
    t = [(0, 0, 0)] * total
    R = [np.empty([3, 3])] * total
    for i in range(total):
        num[i] = int(lines[5 * i])
        rays1[i] = tuple([float(x) for x in lines[5 * i + 1].split()])
        rays2[i] = tuple([float(x) for x in lines[5 * i + 2].split()])
        t[i] = tuple([float(x) for x in lines[5 * i + 3].split()])
        q = [float(x) for x in lines[5 * i + 4].split()]
        R[i] = quat.as_rotation_matrix(np.quaternion(q[3], q[0], q[1], q[2]))

    redraw()
    plt.show()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
