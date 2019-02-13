import matplotlib.pyplot as plt
import numpy as np

f = open('pure_disp_err.txt')
val = np.array([float(x) for x in f.readline().split()])
arr = [(val[i],     val[i + 1], 
        val[i + 2], val[i + 3],
        val[i + 4]) for i in range(0, len(val), 5)]

#  print(arr[0:10])
print(len(arr))
out = []
d_out = []
sz = []
step = 0.5
expect = np.arange(5.0, 14.0, 0.5)
for mid in expect:
    errs = [x[2] for x in arr if (x[1] >= mid - step / 2 and x[1] <= mid + step / 2)]
    d_errs = [abs(x[3] - x[4]) for x in arr 
              if (x[1] >= mid - step / 2 and x[1] <= mid + step / 2)]
    errs.sort()
    d_errs.sort()
    r = -1 if len(errs) == 0 else errs[int(len(errs) * 0.25)] 
    d_r = -1 if len(d_errs) == 0 else d_errs[int(len(d_errs) * 0.25)] 
    out.append(r)
    d_out.append(d_r)
    sz.append(len(errs))


print(out)
print(d_out)
print(sz)

plt.plot(expect, out, '-o')
plt.xlabel('$\sigma_{predict}$ (in pixels)')
plt.ylabel('$\sigma_{actual}$ (in pixels)')
plt.show()
