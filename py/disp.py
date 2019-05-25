import sys
import operator as op
import matplotlib.pyplot as plt
import numpy as np

min_expected_err = 3.2
quantile = 0.5
points_shown = 20
pred_quantile = 0.95

class EpiErr:
    def __init__(self, data):
        self.disparity = data[0]
        self.expected_err = data[1]
        self.real_err_bef = data[2]
        self.real_err = data[3]
        self.depth_GT = data[4]
        self.depth = data[5]
        self.depth_bef = data[6]
        self.e_bef = data[7]
        self.e_aft = data[8]
episize = 9

if len(sys.argv) != 2:
    print('wanted only file with errors')
    exit(1)

f = open(sys.argv[1])
val = np.array([float(x) for x in f.readline().split()])
arr = [EpiErr(tuple((val[i + j] for j in range(episize)))) for i in range(0, len(val), episize)]

total_points = len(arr)
print(total_points)

arr.sort(key=op.attrgetter('expected_err'))
del arr[int(len(arr) * pred_quantile):]

expect = []
out = []
out_bef = []
d_out = []
d_bef_out = []
sz = []

if points_shown == 1:
    print("one point is not a graph!")
    exit(1)

chunk_size = len(arr) // points_shown

for i in range(points_shown):
    first = i * chunk_size
    last = (i + 1) * chunk_size - 1
    errs = [x.real_err for x in arr[first:last]]
    errs_bef = [x.real_err_bef for x in arr[first:last]] 
    d_errs = [abs(x.depth - x.depth_GT) for x in arr[first:last]]
    d_errs_bef = [abs(x.depth_bef - x.depth_GT) for x in arr[first:last]]
    errs.sort()
    errs_bef.sort()
    d_errs.sort()
    d_errs_bef.sort()
    r = 0 if len(errs) == 0 else errs[int(len(errs) * quantile)] 
    r_bef = 0 if len(errs_bef) == 0 else errs_bef[int(len(errs_bef) * quantile)] 
    d_r = 0 if len(d_errs) == 0 else d_errs[int(len(d_errs) * quantile)] 
    d_r_bef = 0 if len(d_errs_bef) == 0 else d_errs_bef[int(len(d_errs_bef) * quantile)] 
    expect.append((arr[first].expected_err + arr[last].expected_err) / 2)
    out.append(r)
    out_bef.append(r_bef)
    d_out.append(d_r)
    d_bef_out.append(d_r_bef)
    sz.append(len(errs))

print(out)
print(d_out)
print(sz)

fig, ax = plt.subplots(2, 1, sharex=True)

ax[0].plot(expect, out_bef, '-o', label="without subpixel refinement")
ax[0].plot(expect, out, '-o', label="with subpixel refinement")
ax[0].set_ylim(bottom=0)
ax[0].set_xlabel('$\sigma_{predict}$ (in pixels)')
ax[0].set_ylabel('$\sigma_{actual}$ (in pixels)')
ax[0].set_title('disparities')
ax[0].legend()

ax[1].plot(expect, d_bef_out, '-o', label="without subpixel refinement")
ax[1].plot(expect, d_out, '-o', label="with subpixel refinement")
ax[1].set_xlabel('$\sigma_{predict}$ (in pixels)')
ax[1].set_ylabel('d (in meters)')
ax[1].set_title('depths')
ax[1].legend()

plt.show()
