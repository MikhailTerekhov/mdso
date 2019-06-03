import sys
import os
import argparse
from bisect import bisect
import numpy as np
from numpy.linalg import norm
import sympy as sp
import quaternion as quat
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(sys.path[0])),
                             'thirdparty/Sophus/py'))
from sophus.so3 import So3
from sophus.se3 import Se3
from sophus.quaternion import Quaternion as Q
import  matplotlib.pyplot as plt

total_frames = 2500
num_ranges = 5
range_scale = 2
min_range_part = 0.005
quantile = 0.95

def t(motion):
    return motion[0:3,3]

def R(motion):
    return motion[0:3,0:3]

def inverse(motion):
    res = np.zeros((4,4))
    Rt = np.transpose(R(motion))
    res[0:3,0:3] = Rt
    res[0:3,3] = -np.matmul(Rt, t(motion))
    res[3,3] = 1
    return res
        
def angle(R_mat):
    cos_angle = (np.trace(R_mat) - 1) / 2
    if cos_angle < -1:
        cos_angle = -1
    if cos_angle > 1:
        cos_angle = 1
    return np.arccos(cos_angle)

def read_se3(line):
    mat3x4 = np.array([float(s) for s in line.split()]).reshape((3, 4))
    return np.vstack((mat3x4, [0, 0, 0, 1]))

if len(sys.argv) != 2:
    print("I am expecting only the name of the output directory of DSO\n")
    exit(1)

out_dir = sys.argv[1]

traj_f = open(os.path.join(out_dir, 'tracked_frame_to_world.txt'))
traj_GT_f = open(os.path.join(out_dir, 'matrix_form_GT_pose.txt'))

frame_to_world = [read_se3(line) for line in traj_f.readlines()]
frame_to_world_GT = [read_se3(line) for line in traj_GT_f.readlines()]

frame_to_world = frame_to_world[1:]
frame_to_world_GT = frame_to_world_GT[1:]

world_to_frame = [inverse(x) for x in frame_to_world]
world_to_frame_GT = [inverse(x) for x in frame_to_world_GT]

if len(frame_to_world) != len(frame_to_world_GT):
    print(f"I expect trajectory and GT trajectory to be of equal length, "
          "but they are {len(frame_to_world)} and {len(frame_to_world_GT)}")
    exit(2)

print(f'num of frames = {len(frame_to_world)}')

cum_path = [0]
cum_path_GT = [0]
for i in range(1, len(frame_to_world)):
    i_m_1_to_i = np.matmul(world_to_frame[i], frame_to_world[i - 1])
    i_m_1_to_i_GT = np.matmul(world_to_frame_GT[i], frame_to_world_GT[i - 1])
    cum_path.append(cum_path[i - 1] + norm(t(i_m_1_to_i)))
    cum_path_GT.append(cum_path_GT[i - 1] + norm(t(i_m_1_to_i_GT)))
print(f'nframes={len(world_to_frame)}, ncum={len(cum_path)}')

lengths_avail = cum_path_GT[-1] * (1 - min_range_part)
total_parts = (range_scale ** num_ranges - 1) // (range_scale - 1)
range_lengths  = [lengths_avail / total_parts]
for i in range(1, num_ranges + 1):
    range_lengths.append(range_lengths[-1] * range_scale)
print(f'range lengths = {range_lengths}')


trans_drift = [[] for _ in range(len(range_lengths) - 1)]
rot_drift   = [[] for _ in range(len(range_lengths) - 1)]
scale_drift = [[] for _ in range(len(range_lengths) - 1)]

print('start measuring')
total_subseq = len(frame_to_world) * (len(frame_to_world) - 1) // 2
cur_processed = 0
to_be_printed = 1
print_percent = list(range(0, 120, 20))
print('[0% ... ', end='', flush=True)
for fst in range(len(frame_to_world)):
    for lst in range(fst + 1, len(frame_to_world)):
        if cur_processed / total_subseq >= print_percent[to_be_printed] / 100:
            print(f'{print_percent[to_be_printed]}% ... ', end='', flush=True)
            to_be_printed += 1

        fst_to_lst = np.matmul(world_to_frame[lst], frame_to_world[fst])
        fst_to_lst_GT = np.matmul(world_to_frame_GT[lst], frame_to_world_GT[fst])

        fst_to_nxt = np.matmul(world_to_frame[fst + 1], frame_to_world[fst])
        fst_to_nxt_GT = np.matmul(world_to_frame_GT[fst + 1], frame_to_world_GT[fst])

        scale_fix = norm(t(fst_to_nxt_GT)) / norm(t(fst_to_nxt))

        path_length = scale_fix * (cum_path[lst] - cum_path[fst])
        path_length_GT = cum_path_GT[lst] - cum_path_GT[fst]
        
        range_ind = bisect(range_lengths, path_length_GT) - 1
        if range_ind == -1:
            continue
        if range_ind == num_ranges:
            print('range bigger than all the path')
            continue
        
        trans_drift[range_ind].append(100 * norm(scale_fix * t(fst_to_lst) - t(fst_to_lst_GT))
                                            / path_length_GT)

        rot_drift[range_ind].append((180 / np.pi) * angle(np.matmul(np.transpose(R(fst_to_lst)),
                                            R(fst_to_lst_GT))) / path_length_GT)
        scale_drift[range_ind].append(100 * abs(path_length / path_length_GT - 1) 
                                            / path_length_GT)
        cur_processed += 1
print('100%]')

for x in [trans_drift, rot_drift, scale_drift]:
    for i in range(num_ranges):
        x[i].sort()
        x[i] = x[i][:int(quantile * len(x[i]))]
    

print([len(x) for x in trans_drift])
trans_drift_avg = [np.average(x) for x in trans_drift]
rot_drift_avg = [np.average(x) for x in rot_drift]
scale_drift_avg = [np.average(x) for x in scale_drift]

print(f"translational drift = {np.average(trans_drift_avg):.3} %")
print(f"rotational drift    = {np.average(rot_drift_avg):.3} deg/m")
print(f"scale drift    = {np.average(scale_drift_avg):.3} %/m")

print(f'trans drifts = {trans_drift_avg}')
print(f'rot drifts   = {rot_drift_avg}')
print(f'scale drifts = {scale_drift_avg}')

fig, (ax_trans, ax_rot, ax_scale) = plt.subplots(3, 1)
ax_trans.set_title('translational error distributions (%)')
ax_trans.set_xticklabels([])
ax_trans.violinplot(trans_drift, showmeans=True)

ax_rot.set_title('rotational error distributions (deg/m)')
ax_rot.set_xticklabels([])
ax_rot.violinplot(rot_drift, showmeans=True)

ax_scale.set_title('scale error distributions (%/m)')
ax_scale.set_xlabel('ranges of lengths of subsequences (m)')
ax_scale.set_xticklabels([''] + [f'[{range_lengths[i]:.1f}, {range_lengths[i+1]:.1f})'
                                for i in range(num_ranges)])
ax_scale.violinplot(scale_drift, showmeans=True)

plt.show()
