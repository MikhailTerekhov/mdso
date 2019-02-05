import matplotlib.pyplot as plt
import numpy as np

f = open('pure_disp_err.txt')
val = np.array([float(x) for x in f.readline().split()])
arr = [(val[i],     val[i + 1], 
        val[i + 2], val[i + 3],
        val[i + 4]) for i in range(0, len(val), 5)]

pred = [x[1] for x in arr]
real = [x[2] for x in arr]

plt.scatter(pred, real, s=0.2)
plt.xlim(4, 15)
plt.ylim(0, 6)
plt.show()
