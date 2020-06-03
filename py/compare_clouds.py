import numpy as np
import scipy
import argparse
from pathlib import Path
from traj.traj import *
from bisect import bisect_left
import progressbar
import seaborn as sns
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
import sys

parser = argparse.ArgumentParser()
parser.add_argument('cloud', help='Cloud from odometry')
parser.add_argument('gt', help='GT cloud')
parser.add_argument('-d', '--distance', type=float, nargs='?', default=0.002,
                    help='Closest distance')
args = parser.parse_args()

def read_cloud(fname):
    f = open(fname, 'rb')
    cloud = PlyData.read(f).elements[0]
    points = np.vstack([cloud.data['x'], cloud.data['y'], cloud.data['z']]).T
    #  colors = np.hstack([cloud.data['red'], cloud.data['green'],
                        #  cloud.data['blue']])
    return points

def closeness(base, ref):
    tree = scipy.spatial.KDTree(base)
    close_num = 0
    for pi in progressbar.progressbar(range(ref.shape[0])):
        p = ref[pi,:]
        d, i = tree.query(p)
        if d < args.distance:
            close_num += 1
    return close_num / ref.shape[0]


vo_points = read_cloud(args.cloud)
gt_points = read_cloud(args.gt)

print(f'completeness = {closeness(vo_points, gt_points)}')
print(f'precision = {closeness(gt_points, vo_points)}')
