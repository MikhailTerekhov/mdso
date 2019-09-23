import sys
import os
import argparse

import numpy as np
import pandas as pd
import quaternion as quat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import norm

arr_len = 0.05

# workaround for Axes3D aspect ratio to be 1
# found on https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
def set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)

def to_motion(ser):
    x = ser.tolist()
    q = np.quaternion(x[3], x[0], x[1], x[2])
    R = quat.as_rotation_matrix(q)
    t = np.array(x[-3:])
    return (t, R)

def extract_motions(fname):

    tbl = pd.read_csv(fname, sep='\s+', header=None,
                      index_col=0, dtype=np.float64)
    ser = tbl.apply(to_motion, axis=1)
    fnums = ser.index.values.tolist()
    motions = ser.tolist()
    return list(zip(fnums, motions))


def draw_arrowed(axes, motions, color, label):
    centers = np.array([-np.matmul(mot[1].T, mot[0]) for mot in motions])
    dir_vects = np.array([mot[1][2, :] for mot in motions])

    axes.quiver(centers[:, 0], centers[:, 1], centers[:, 2],
              dir_vects[:, 0], dir_vects[:, 1], dir_vects[:, 2], 
                color=color, normalize=True, arrow_length_ratio=0.2, 
                length=arr_len, label=label)

def draw_track(axes, motions, color, label):
    centers = np.array([-np.matmul(mot[1].T, mot[0]) for mot in motions])
    axes.plot(centers[:, 0], centers[:, 1], centers[:, 2], color=color, label=label)

def select(indexed, first, last):
    return [val for (ind, val) in indexed 
                  if ind >= first and ind <= last]


parser = argparse.ArgumentParser()
parser.add_argument("dir", help="DSO output directory")
parser.add_argument("-r", "--russian", help="use russian legend",
                    action="store_true")
parser.add_argument("-q", "--quiver", help="plot camera directions"
                    " (if not specified, only draw position curve)", 
                    action="store_true")
parser.add_argument("-f", "--first", help="truncate frames with nums less than this", 
                    type=int)
parser.add_argument("-l", "--last", help="truncate frames with nums bigger than this", 
                    type=int)
args = parser.parse_args()

out_dir = args.dir

if (out_dir[-1] == '/'):
    out_dir = out_dir[0:-1]

actual = extract_motions(out_dir + '/tracked_pos.txt')

pred_path = out_dir + '/predicted_pos.txt'
has_predicted = os.path.isfile(pred_path)
if has_predicted:
    predicted = extract_motions(pred_path)
else:
    print('no prediction provided')

sm_path = out_dir + '/matched_pos.txt'
has_stereo_matched = os.path.isfile(sm_path)
if has_stereo_matched:
    stereo_matched = extract_motions(sm_path)
else:
    print('no stereo-matching provided')

gt_path = out_dir + '/ground_truth_pos.txt'
has_ground_truth = os.path.exists(gt_path)
if has_ground_truth:
    ground_truth = extract_motions(gt_path)
else:
    print('no ground truth provided')

first = 0 if args.first is None else args.first
last = np.inf if args.last is None else args.last;

print(f'f={first} l={last}')
print(f'act elem={actual[0]}')
actual = select(actual, first, last)
print(f'total selected = {len(actual)}')

if has_predicted:
    predicted = select(predicted, first, last)
if has_stereo_matched:
    stereo_matched = select(stereo_matched, first, last)
if has_ground_truth:
    ground_truth = select(ground_truth, first, last)


fig = plt.figure()
ax = Axes3D(fig)
ax.set_aspect('equal')

ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.set_yticks([])

ax.view_init(azim=-90, elev=0)

draw_proc = draw_arrowed if args.quiver else draw_track

if args.russian:
    label_ours   = 'полученная оценка'
    label_prec   = 'точная траектория'
    label_stereo = 'ORB + five point'
else:
    label_ours   = 'dso'
    label_prec   = 'ground truth'
    label_stereo = 'ORB + five point'


draw_proc(ax, actual, 'orange', label_ours)
#  if has_predicted:
    #  draw_motions(ax, predicted, 'blue', 'предсказанная траектория')
if has_ground_truth:
    draw_proc(ax, ground_truth, 'green', label_prec)
elif has_stereo_matched:
    draw_proc(ax, stereo_matched, 'green', label_stereo)

#  ax.scatter3D([0], [0], [0], color='black', label='базовый кадр')

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.3), fontsize='large')

set_axes_equal(ax)
plt.show()
