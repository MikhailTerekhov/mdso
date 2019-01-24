import numpy as np

f = open('disp_err.txt')
val = np.array([float(x) for x in f.readline().split()])
arr = [(val[i],     val[i + 1], 
        val[i + 2], val[i + 3],
        val[i + 4]) for i in range(0, len(val), 5)]

#  print(arr[0:10])
print(len(arr))
out = []
sz = []
for up in range(5, 15):
    errs = [x[2] for x in arr if (x[1] >= up - 1 and x[1] <= up)]
    errs.sort()
    r = -1 if len(errs) == 0 else errs[int(len(errs) * 0.4)] 
    out.append(r)
    sz.append(len(errs))

print(out)
print(sz)
