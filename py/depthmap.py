import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

_, ax = plt.subplots()

f = open('test/data/MultiFoV/data/depth/img0001_0.depth')
if not f.readable():
    print('file not found')
    exit(0)
dp = np.array([float(s) for s in f.readline().split()])

sr = np.copy(dp)
sr.sort()
prob = np.linspace(0., 1., num=len(sr))
ax.plot(sr, prob)
e = sr[int(0.85 * sr.size)]
plt.xlim(0, e)
#  print(f'b={b} e={e}')
#  colmap = cm.get_cmap('plasma', 256)

#  dp.shape = (480,640)
#  print(dp.dtype.name)
#  print(dp[0, 0])
#  im = ax.imshow(dp, cmap=colmap, vmin=b, vmax=e)

#  plt.axis('off')
#  plt.colorbar(im)
plt.show()
