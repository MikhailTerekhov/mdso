import numpy as np
from numpy.linalg import norm

class se3:
    def __init__(self, R, t):
        self.R = R
        self.t = t

    def inverse(self):
        return se3(self.R.T, -np.matmul(self.R.T, self.t))
    
    def __mul__(self, other):
        return se3(np.matmul(self.R, other.R), np.matmul(self.R, other.t) + self.t)

arr_len = 0.05

def to_motion(s):
    vals = np.array([float(x) for x in s.split()])
    mat3x4 = vals.reshape((3, 4))
    return se3(mat3x4[:,:3], mat3x4[:,3])

def extract_motions(fname):
    f = open(fname)
    return [to_motion(s) for s in f.readlines()]


def align_to_zero(traj):
    fix = traj[0].inverse()
    return [fix * m for m in traj]

def align(traj, gt):
    t01 = traj[1].inverse() * traj[0]
    g01 = gt[1].inverse() * gt[0]
    tg = gt[0].inverse() * traj[0]
    res = [tg * m for m in traj]
    scale_fix = norm(g01.t) / norm(t01.t)
    res = [se3(m.R, scale_fix * m.t) for m in res]
    return res


