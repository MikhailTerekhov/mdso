import sys
import matplotlib.pyplot as plt
import numpy as np

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

#  print(arr[0:10])
print(len(arr))
out = []
out_bef = []
d_out = []
d_bef_out = []
sz = []
step = 0.5
expect = np.arange(4.0, 10.0, 0.25)
for mid in expect:
    errs = [x.real_err for x in arr 
            if (x.expected_err >= mid - step / 2 
                and x.expected_err <= mid + step / 2)]
    errs_bef = [x.real_err_bef for x in arr 
            if (x.expected_err >= mid - step / 2 
                and x.expected_err <= mid + step / 2)]
    d_errs = [abs(x.depth - x.depth_GT) for x in arr 
              if (x.expected_err >= mid - step / 2 
                  and x.expected_err <= mid + step / 2)]
    d_errs_bef = [abs(x.depth_bef - x.depth_GT) for x in arr 
              if (x.expected_err >= mid - step / 2 
                  and x.expected_err <= mid + step / 2)]
    errs.sort()
    errs_bef.sort()
    d_errs.sort()
    d_errs_bef.sort()
    r = -1 if len(errs) == 0 else errs[int(len(errs) * 0.5)] 
    r_bef = -1 if len(errs_bef) == 0 else errs_bef[int(len(errs_bef) * 0.5)] 
    d_r = -1 if len(d_errs) == 0 else d_errs[int(len(d_errs) * 0.5)] 
    d_r_bef = -1 if len(d_errs_bef) == 0 else d_errs_bef[int(len(d_errs_bef) * 0.5)] 
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
