import sys
import os
from fnmatch import fnmatch
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from traj import *
from itertools import cycle
from bisect import bisect_left, bisect_right


alpha = 1
win_size = 10
img_w = 12
img_h = 12

sns.reset_orig()
def color_gen(num):
    return cycle(sns.color_palette())
    #  return cycle(['#33A02C', '#FC8D62', '#a860b3'])

def set_invis(a):
    for x in a:
        x.set_visible(False)

def set_vis(a):
    for x in a:
        x.set_visible(True)

def draw_arrowed(axes, motions, color, label, direction='z'):
    centers = np.array([mot.t for mot in motions])
    cols = {'x':0, 'y':1, 'z':2}
    dir_vects = np.array([mot.R[:, cols[direction]] for mot in motions])

    arr_len = np.median([norm(centers[i + 1] - centers[i]) 
                         for i in range(centers.shape[0] - 1)]) / 2

    return axes.quiver(centers[:, 0], centers[:, 1], centers[:, 2],
              dir_vects[:, 0], dir_vects[:, 1], dir_vects[:, 2], 
                color=color, normalize=True, arrow_length_ratio=0.2, 
                length=arr_len, label=label)

def draw_track(axes, motions, color, label, direction='z'):
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


parser = argparse.ArgumentParser()
parser.add_argument("traj", help="one or more files with trajectories "
                    "(expects them to reside in one directory)", nargs='+')
parser.add_argument("--gt", help="file with ground truth trajectory")
parser.add_argument("-a", "--align", help="Align GT trajectory?",
                    action="store_true")
parser.add_argument("-s", "--scale_fix", help="Try to align the scale with ground truth?",
                    action="store_true")
parser.add_argument("--video_dir", help="directory to put video with sliding "
                    "window of trajectories into")
parser.add_argument("-r", "--russian", help="use russian legend",
                    action="store_true")
parser.add_argument("-q", "--quiver", help="plot camera directions"
                    " (if not specified, only draw position curve)", 
                    action="store_true")
parser.add_argument("-d", "--direction", help="Direction axis of arrows."
                    " Only useful when quiver is on.", 
                    default='z', const='z', nargs='?',
                    choices=['x', 'y', 'z'])
parser.add_argument("-f", "--first", help="truncate frames with nums less than this", 
                    type=int)
parser.add_argument("-l", "--last", help="truncate frames with nums bigger than this", 
                    type=int)
parser.add_argument("-p", "--shown_plane", help="The plain that is shown in the "
                    "default orientation. Could be xy, xz or yz.", default="xz")
parser.add_argument("--labels", help="labels for trajectories, excluding"
                    "ground truth", nargs='+')
parser.add_argument("-t", "--timestamps", help="File with timestamps of the poses "
                    "provided in trajectories. If set, --first and --last flags "
                    "are considered to be timestamps rather than frame numbers")
parser.add_argument("-e", "--errors", help="plot errors of each trajectory. "
                    "Only works if ground truth is provided", action="store_true")
parser.add_argument("--axis_errors", help="plot errors for each trajectory and "
                    "each axis independently. Only works when --errors is specified",
                    action="store_true")
parser.add_argument("--rot_errors", help="plot rotational errors for each trajectory."
                    "Only works when --errors is specified", action="store_true")
args = parser.parse_args()

if args.russian:
    xlabel     = 'Ось X (м)'
    ylabel     = 'Ось Y (м)'
    zlabel     = 'Ось Z (м)'
    label_gt   = 'Точная траектория'
    label_zerr = 'Ошибка по оси Z, м'
    label_time = 'Время, с'
else:
    xlabel     = 'X axis (m)'
    ylabel     = 'Y axis (m)'
    zlabel     = 'Z axis (m)'
    label_gt   = 'Ground truth'
    label_zerr = 'Z axis error, m'
    label_time = 'Time, s'

if args.timestamps is not None:
    timestamps = np.loadtxt(args.timestamps, dtype=np.int64)

def crop_roi(traj):
    global args
    f = 0
    l = len(traj)
    if args.timestamps is not None:
        print(f'ts={timestamps}')
        if args.first is not None:
            f = bisect_left(timestamps, args.first)
        if args.last is not None:
            l = bisect_right(timestamps, args.last)
    else:
        if args.first is not None:
            f = args.first
        if args.last is not None:
            l = args.last
    f = max(f, 0)
    l = min(l, len(traj))
    print(f'f={f} l={l}')
    return traj[f:l]

if args.labels and len(args.traj) != len(args.labels):
    print('number of labels must be equal to the number of trajectories')
    exit(0)

fig = plt.figure(figsize=(img_w, img_h))
ax = Axes3D(fig)

ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

tick_pad = 8
ax.xaxis.set_tick_params(pad=tick_pad)
ax.yaxis.set_tick_params(pad=tick_pad)
ax.zaxis.set_tick_params(pad=tick_pad)

#  ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
labelpad = 15
if args.shown_plane == "xy":
    ax.view_init(azim=-90, elev=-90)
    ax.set_zticks([])
    ax.set_xlabel(xlabel, labelpad=labelpad)
    ax.set_ylabel(ylabel, labelpad=labelpad)
elif args.shown_plane == "xz":
    ax.set_yticks([])
    ax.set_xlabel(xlabel, labelpad=labelpad)
    ax.set_zlabel(zlabel, labelpad=labelpad)
    ax.view_init(azim=-90, elev=0)
elif args.shown_plane == "yz":
    ax.set_xticks([])
    ax.set_ylabel(ylabel, labelpad=labelpad)
    ax.set_zlabel(zlabel, labelpad=labelpad)
    ax.view_init(azim=180, elev=0)
else:
    print(f'unsupported value {args.shown_plane} for shown_pane')


draw_proc = draw_arrowed if args.quiver else draw_track

traj_word = 'траектория' if args.russian else 'trajectory'
labels = args.labels if args.labels \
         else [f'{traj_word} #{i}' for i in range(len(args.traj))]

if args.gt:
    ground_truth = extract_motions(args.gt)
    ground_truth = crop_roi(ground_truth)
    if args.align:
        ground_truth = align_to_zero(ground_truth)
    has_gt = True
else:
    print('no ground truth provided')
    has_gt = False


artists = []
trajs = []

colors = color_gen(len(args.traj) + 1)

mpl.rcParams.update({'font.size': 18})

for ind, (fname, label) in enumerate(zip(args.traj, labels)):
    print(f'processing {fname}...')
    traj = extract_motions(fname)

    print(f'full trajectory size is {len(traj)}\n')
    traj = crop_roi(traj)
    print(f'cropped trajectory size is {len(traj)}\n')

    if len(traj) < 2:
        print(f'too small num of positions: {len(traj)}')
        continue

    if has_gt and args.align:
        print(f'align, len={len(traj)} {len(ground_truth)}')
        traj = align(traj, ground_truth, need_scale_fix=args.scale_fix)

    col = next(colors)

    artists.append(draw_proc(ax, traj, mpl.colors.to_rgba(col, alpha=alpha),
                            label, direction=args.direction))
    trajs.append(traj)

#  next(colors)
if has_gt:
    draw_proc(ax, ground_truth, next(colors), label_gt, direction=args.direction)

ax.legend(loc='lower left', bbox_to_anchor=(0.5, 0.6), fontsize='medium')

set_axes_equal(ax)

if args.timestamps is not None:
    timestamps = crop_roi(timestamps)

if args.video_dir:
    os.makedirs(args.video_dir, exist_ok=True)
    for a in artists[min(win_size,len(artists)):]:
        set_invis(a)
    fig.savefig(os.path.join(args.video_dir, '0.png'))
    for i in range(len(artists) - win_size):
        set_invis(artists[i])
        set_vis(artists[i + win_size])
        #  ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.3), fontsize='large')
        fig.savefig(os.path.join(args.video_dir, f'{i + 1}.png'))



def axis_errs(m1, m2):
    return [abs(v) for v in m1.t - m2.t]

def err(m1, m2):
    return norm(m1.t - m2.t)

def rot_err(m1, m2):
    return (m1 * m2.inverse()).angle() * 180 / np.pi

def azim_err(m1, m2):
    dir1 = m1.R[:,0].reshape(3)
    dir2 = m2.R[:,0].reshape(3)
    dir1 = dir1[0:2]
    dir2 = dir2[0:2]
    dir1 /= norm(dir1)
    dir2 /= norm(dir2)
    sina = np.clip(np.cross(dir1, dir2), -1, 1)
    return 180. / np.pi * np.arcsin(sina)

if args.errors:
    assert args.gt

    plt.figure()
    if args.axis_errors:
        assert len(trajs) == 2
        assert args.timestamps
        ts = timestamps / 30
        colors = color_gen(3)
        ax_names = ['x', 'y', 'z']
        for traj_ind, (traj, label) in enumerate(zip(trajs, labels)):
            errors = np.array([axis_errs(t, g) for t, g in zip(ground_truth, traj)])
            plt.plot(ts, errors[:, 2], label=label, color=next(colors))
        plt.xlabel(label_time, labelpad=-15)
        plt.ylabel(label_zerr)
    else:
        colors = color_gen(len(args.traj))
        for traj, label in zip(trajs, labels):
            errors = [err(t, g) for t, g in zip(ground_truth, traj)]
            plt.plot(errors, label=label, color=next(colors))

    
    if args.rot_errors:
        plt.figure()
        colors = color_gen(len(args.traj))
        plt.title('Азимутальные ошибки (°)')
        for traj, label in zip(trajs, labels):
            errors = [azim_err(t, g) for t, g in zip(traj, ground_truth)]
            plt.plot(errors, label=label, color=next(colors))


    plt.legend()

plt.show()
