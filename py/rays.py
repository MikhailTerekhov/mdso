from operator import add
import numpy as np
import pandas as pd
import quaternion as quat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection


f = open('log.txt')
rays = 2 * [(0, 0, 0)]
for i in range(2):
    rays[i] = np.array(list(map(float, f.readline().split())))
t = np.array(list(map(float, f.readline().split())))
q = list(map(float, f.readline().split()))
q = (q[3], q[0], q[1], q[2])
R = quat.as_rotation_matrix(np.quaternion(*q))

print(R)
print(np.linalg.det(R))

lines = [None] * 2
rays[0] *= 3
lines[0] = np.column_stack((np.array([0, 0, 0]), np.array(rays[0])))
start2 = np.matmul(R.T, -t)
dir2 = np.matmul(R.T, rays[1])
print(dir2)
lines[1] = np.column_stack((start2, start2 + dir2))


fig = plt.figure()
ax = Axes3D(fig)

print(lines[1].T.tolist())
ax.plot(*lines[0].tolist(), color='green')
ax.plot(*lines[1].tolist(), color='orange')
ax.scatter(*lines[0][:,0])
ax.scatter(*lines[1][:,0])

plane1 = list(map(tuple, [lines[0][:,0], lines[1][:,0], lines[0][:,1]]))
plane2 = list(map(tuple, [lines[1][:,0], lines[0][:,0], lines[1][:,1]]))
polycol = Poly3DCollection([plane1, plane2])
polycol.set_facecolor((0,0,1,0.4))
print(plane1)
ax.add_collection3d(polycol)

#  draw_motions(ax, actual, 'orange')
#  draw_motions(ax, predicted, 'blue')n.

#  verts = [[(0, 0, 0), (0, 0, 1), (0, 1, 0)], [(1, 0, 0), (1, 0, 1), (1, 1, 0)]]
#  col = Poly3DCollection(verts, linewidth=1, edgecolors='k')
#  col.set_facecolor((0,0,1,0.1))

#  ax.add_collection3d(col)
plt.show()
