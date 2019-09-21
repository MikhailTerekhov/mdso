import sys
import os
from fnmatch import fnmatch
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from traj import *


alpha = 0.5
win_size = 10
img_w = 12
img_h = 12


def set_invis(a):
    for x in a:
        x.set_visible(False)

def set_vis(a):
    for x in a:
        x.set_visible(True)

def draw_arrowed(axes, motions, color, label):
    centers = np.array([mot.t for mot in motions])
    dir_vects = np.array([mot.R[:, 2] for mot in motions])

    return axes.quiver(centers[:, 0], centers[:, 1], centers[:, 2],
              dir_vects[:, 0], dir_vects[:, 1], dir_vects[:, 2], 
                color=color, normalize=True, arrow_length_ratio=0.2, 
                length=0.5, label=label)

def draw_track(axes, motions, color, label):
    centers = np.array([mot.t for mot in motions])
    return axes.plot(centers[:, 0], centers[:, 1], centers[:, 2],
                     color=color, label=label)

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

fig = plt.figure(figsize=(img_w, img_h))
ax = Axes3D(fig)
ax.set_aspect('equal')

ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.set_yticks([])
ax.view_init(azim=-90, elev=0)


parser = argparse.ArgumentParser()
parser.add_argument("--gt", help="file with ground truth trajectory")
parser.add_argument("--video_dir", help="directory to put video with sliding "
                    "window of trajectories into")
parser.add_argument("-r", "--russian", help="use russian legend",
                    action="store_true")
parser.add_argument("-q", "--quiver", help="plot camera directions"
                    " (if not specified, only draw position curve)", 
                    action="store_true")
parser.add_argument("-f", "--first", help="truncate frames with nums less than this", 
                    type=int)
parser.add_argument("-l", "--last", help="truncate frames with nums bigger than this", 
                    type=int)
parser.add_argument("traj", help="one or more files with trajectories "
                    "(expects them to reside in one directory)", nargs='+')
args = parser.parse_args()


draw_proc = draw_arrowed if args.quiver else draw_track
label_ours = 'полученная оценка' if args.russian else 'dso'
label_prec = 'точная траектория' if args.russian else 'ground_truth'


if args.gt:
    ground_truth = extract_motions(args.gt)
    firstgt = 0 if args.first is None else args.first
    lastgt  = len(ground_truth) if args.last is None else args.last;
    ground_truth = ground_truth[firstgt:lastgt]
    ground_truth = align_to_zero(ground_truth)
    draw_proc(ax, ground_truth, 'green', label_prec)
    has_gt = True
else:
    print('no ground truth provided')
    has_gt = False


artists = []

for fname in args.traj:
    print(f'processing {fname}...')
    traj = extract_motions(fname)
    if len(traj) < 2:
        print(f'too small num of positions: {len(traj)}')
        continue

    first = 0 if args.first is None else args.first
    last = len(traj) if args.last is None else args.last;

    traj = traj[first:last]
    if has_gt:
        traj = align(traj, ground_truth)
    artists.append(draw_proc(ax, traj, mpl.colors.to_rgba('orange', alpha=alpha),
                            label_ours))

set_axes_equal(ax)

if args.video_dir:
    os.makedirs(args.video_dir, exist_ok=True)
    for a in artists[min(win_size,len(artists)):]:
        set_invis(a)
    #  plt.show()
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.3), fontsize='large')
    fig.savefig(os.path.join(args.video_dir, '0.png'))
    for i in range(len(artists) - win_size):
        set_invis(artists[i])
        set_vis(artists[i + win_size])
        #  ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.3), fontsize='large')
        fig.savefig(os.path.join(args.video_dir, f'{i + 1}.png'))
else:
    plt.show();
