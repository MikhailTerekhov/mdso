import sys
from os.path import join, abspath, dirname
import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("lut", help="Undistortion LUT file from dataset in binary form")
parser.add_argument("out", help="Undistortion LUT output file")
args = parser.parse_args()

lut = np.fromfile(args.lut, np.double)
lut = lut.reshape([2, lut.size // 2])
lut = lut.T[:, 1::-1].T
lut = lut.reshape((2, 1024, 1024))

out_f = open(args.out, 'w')
print(f'{lut.shape[1]} {lut.shape[2]}', file=out_f)
for i in range(lut.shape[1]):
    for j in range(lut.shape[2]):
        print(f'{i} {j} {lut[0,i,j]} {lut[1,i,j]}', file=out_f)
out_f.close()
