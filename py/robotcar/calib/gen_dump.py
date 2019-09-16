import os
import sys
import re
import numpy as np
import cv2
from PIL import Image
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear as demosaic
import argparse
import operator
import random

parser = argparse.ArgumentParser()
parser.add_argument("dump", help="Name of the output file with dump")
parser.add_argument("pattern_files", help="Files with found patterns", 
                    nargs='+')
args = parser.parse_args()

sz = (8, 6)
dumps = []

for num, fname in enumerate(args.pattern_files):
    corners = np.genfromtxt(fname, delimiter=',')

    ts = int(os.path.basename(fname)[:-8])
    dumps.append(([], ts))
    for i in range(sz[0] * sz[1]):
        dumps[-1][0].append((corners[i,0], corners[i,1],
                             i % sz[0], (sz[1] - 1 - i // sz[0]), 0))
    print(f'{num + 1} / {len(args.pattern_files)}')
    

dump_f = open(args.dump, 'w')

print(f'1024 1024 {len(dumps)}', file=dump_f)
for d in dumps:
    d[0].sort(key=operator.itemgetter(2))
    d[0].sort(key=operator.itemgetter(3))
    print(f'{len(d[0])} {d[1]}', file=dump_f)
    for p in d[0]:
        print(*p, file=dump_f)
