import sys
import matplotlib.pyplot as plt
import numpy as np

errs_f = open(sys.argv[1])
vals = np.array([[float(x) for x in l.split()] for l in errs_f.readlines()])

sz = (1024, 1024)

abs_e = np.zeros(sz)
x_e = np.zeros(sz)
y_e = np.zeros(sz)
x_norm_e = np.zeros(sz)
y_norm_e = np.zeros(sz)

for v in vals:
    x, y, ex, ey = v
    norm = np.hypot(ex, ey)
    abs_e[int(y), int(x)] = norm
    x_e[int(y), int(x)] = ex
    y_e[int(y), int(x)] = ey
    x_norm_e[int(y), int(x)] = ex / norm
    y_norm_e[int(y), int(x)] = ey / norm

plt.subplot(221)
plt.imshow(abs_e)
plt.title('absolute error (px)')
plt.colorbar()

plt.subplot(222)
dir_samp = 40
x = list(range(0, sz[0], sz[0] // dir_samp))
y = list(range(0, sz[1], sz[1] // dir_samp))
x, y = np.meshgrid(x, y)
plt.xlim(0 ,sz[1])
plt.ylim(0 ,sz[0])
plt.gca().set_aspect('equal', adjustable='box')
plt.gca().invert_yaxis()
plt.quiver([x], [y], y_norm_e[x, y], x_norm_e[x, y], angles='xy')
plt.title('error direcions')

plt.subplot(223)
plt.imshow(x_e)
plt.title('error along x')
plt.colorbar()

plt.subplot(224)
plt.imshow(y_e)
plt.title('error along y')
plt.colorbar()

plt.show()
