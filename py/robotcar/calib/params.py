#!/usr/bin/python3

class Args:
    def __init__(self):
        pass
    def __repr__(self):
        return f'dump={self.dump} direct={self.direct} fix_p={self.fix_p}\n'  \
               f'zero_skew={self.zero_skew} equal_focal={self.equal_focal}\n' \
               f'poly_pow={self.poly_pow}'

import sys
import numpy as np
import os
from scipy.interpolate import interp2d
import argparse
import matplotlib.pyplot as plt
import operator

parser = argparse.ArgumentParser()
parser.add_argument("calib_dir", help="Name of the root directory of supervised_calibration")
parser.add_argument("given_intrin", help="Given intrinsic parameters of the camera")
parser.add_argument("given_lut", help="Given undistortion LUT of the camera")
parser.add_argument("cam_out", help="Name of the file to output calibrated camera model")
parser.add_argument('-j', '--jobs', help="Number of threads to be spawned", type=int, default=50)
parser.add_argument('-s', '--start', help="Start from this arg bundle in the list",
                    type=int, default=50)
p_args = parser.parse_args()

sys.path.insert(0, os.path.join(p_args.calib_dir, "build/wrappers/"))
sys.path.insert(0, os.path.join(p_args.calib_dir, "utils/python/"))
import super_calibration as sc
from read_file import read_dump
from read_lut import read_lut

from joblib import Parallel, delayed
import multiprocessing


def calc_err(lut, cam, Kinv):
    sumErr = 0
    for y in range(lut.shape[1]):
        for x in range(lut.shape[2]):
            uv = np.array([x, y, 1])
            p = cam.project(np.matmul(Kinv, uv))
            pl = lut[:, y, x][::-1]
            err = np.linalg.norm(p - pl)
            sumErr += err
    return sumErr / (lut.shape[1] * lut.shape[2])

intrin_f = open(p_args.given_intrin)
vals = [float(x) for x in intrin_f.readline().split()]
focal = vals[:2]
calibCenter = vals[2:4]
K = np.array([[focal[0], 0, calibCenter[0]],
             [0, focal[1], calibCenter[1]],
             [0,        0,              1]])
Kinv = np.linalg.inv(K)

lut = read_lut(p_args.given_lut)
xinds = np.arange(lut.shape[1])
yinds = np.arange(lut.shape[2])

center = np.array([interp2d(xinds, yinds, lut[i,:,:])(calibCenter[0], calibCenter[1])[0]
          for i in [1, 0]])

args_lst = []
for dump in ['data/dumps/rear_filtered.txt',
             'data/dumps/rear_old.txt',
             'data/dumps/rear_new.txt',
             'data/dumps/rear_all.txt']:
    for direct in [False, True]:
        for fix_p in [False, True]:
            for zero_skew in [False, True]:
                for equal_focal in [False, True]:
                    for poly_pow in [6, 7, 8]:
                        cur = Args()
                        cur.lut = lut
                        cur.Kinv = Kinv
                        cur.dump = dump
                        cur.direct = direct
                        cur.fix_p = fix_p
                        cur.zero_skew = zero_skew
                        cur.equal_focal = equal_focal
                        cur.poly_pow = poly_pow   
                        cur.return_cam = False
                        args_lst.append(cur)

print(f'test args = {args_lst[0]}')

num_checked = 0
total = len(args_lst)

def check_args(args):
    fps, w, h = read_dump(args.dump)

    if args.direct:
        camAtan = sc.createCamera("Atan-fisheye", w, h, polynom_n=args.poly_pow, 
                                  equal_focal=args.equal_focal, zero_skew=args.zero_skew,
                                  fix_principal=args.fix_p)
        if args.fix_p:
            camAtan.setPrincipal(center)
        error = sc.calibrateCamera(camAtan, fps, with_loss=False)
    else:
        camOmni = sc.createCamera("Omnidirectional", w, h, polynom_n=args.poly_pow, 
                                  equal_focal=args.equal_focal, zero_skew=args.zero_skew,
                                  fix_principal=args.fix_p)
        if args.fix_p:
            camOmni.setPrincipal(center)
        error = sc.calibrateCamera(camOmni, fps)

        camAtan = sc.createCamera("Atan-fisheye", w, h, polynom_n=args.poly_pow, 
                                  equal_focal=args.equal_focal, zero_skew=args.zero_skew,
                                  fix_principal=args.fix_p)
        if args.fix_p:
            camAtan.setPrincipal(center)
        convError = camAtan.convertFromModel(camOmni)

    camAtan.setPrincipal(center)
    sumErr = 0

    global num_checked
    global total
    num_checked += 1
    print(f'{num_checked} / {total}')

    if args.return_cam:
        return camAtan
    else:
        return (calc_err(args.lut, camAtan, args.Kinv), args)

#  print(f'JOBS COUNT = {p_args.jobs}')
#  cams = Parallel(n_jobs=p_args.jobs)(delayed(check_args)(args) for args in args_lst)

start = p_args.start
args_lst = args_lst[start:]
cams = []
out = open('data/rear_all.txt', 'a')
for num, args in enumerate(args_lst):
    num += start
    args.return_cam = True
    cam = check_args(args)
    print(args, file=out)
    err = calc_err(args.lut, cam, args.Kinv)
    print(err, file=out)
    print(f'{num + 1}', file=out)
    out.flush()
    cams.append((err, args))
    print(f'{num + 1}')


#  best = min(cams, key=operator.itemgetter(0))
#  print(f'best cam err = {best[0]}\nargs = {best[1]}', file=out)

#  args = best[1]
#  args.return_cam = True
#  cam = check_args(args)
#  print('resulting cam:')
#  cam.print()
#  cam.saveToFile(p_args.cam_out)
