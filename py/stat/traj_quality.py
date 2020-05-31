import sys
import os
import argparse
from bisect import bisect, bisect_left
import numpy as np
from numpy.linalg import norm
import pandas as pd
import progressbar
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(sys.path[0])),
                             'thirdparty/Sophus/py'))
import seaborn as sns
import matplotlib as mpl
import  matplotlib.pyplot as plt
#  import tikzplotlib

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

def motion_scale(motion, scale):
    result = motion
    result[0:3, 3] *= scale
    return result

def scale_adjusted(traj, scale):
    return [motion_scale(m, scale) for m in traj]

def read_traj_adjusted(fname, scale):
    f = open(fname)
    return scale_adjusted(
        [read_se3(line) for line in f.readlines()], scale)



parser = argparse.ArgumentParser()
parser.add_argument("traj", help="file with trajectory")
parser.add_argument("gt", help="file with ground truth")
parser.add_argument("-f", "--first", help="the first frame to be considered", type=int)
parser.add_argument("--overall_scale", help="scale multiplier for all trajectories",
                    type=float, nargs='?', default=1)
parser.add_argument("--inds", help="file with timestamps")
parser.add_argument("--other", help="file with another trajectory")
parser.add_argument("--other_inds", help="file with another trajectory's timestamps")
parser.add_argument("-l", "--label", help="label of the trajectory", 
                    nargs="?", default="trajectory")
parser.add_argument("-L", "--other_label", help="label of another trajectory",
                    nargs="?", default="other trajectory")
parser.add_argument("-s", "--scale_fix", help="do we need to adjust the scale "
                    "of the trajectory", action="store_true")
parser.add_argument("-S", "--other_scale_fix", help="do we need to adjust the scale "
                    "of another trajectory", action="store_true")
parser.add_argument("-r", "--russian", help="labels in russian", action="store_true")
args = parser.parse_args()

if args.russian:
    label_trans    = 'Распределения ошибки в смещении (%)'
    label_rot      = 'Распределения ошибки во вращении (°/м)'
    label_scale    = 'Распределения ошибки в масштабе (%/м)'
    label_ranges   = 'Промежутки длин подпоследовательностей (м)'
else:
    label_trans    = 'Translational error distributions (%)'
    label_rot      = 'Rotational error distributions (°/m)'
    label_scale    = 'Scale error distributions (%/m)'
    label_ranges   = 'Ranges of lengths of subsequences (m)'

def get_errors(traj_fname, inds_fname, gt_fname, first, label, scale_fix):
    frame_to_world_GT = read_traj_adjusted(gt_fname, args.overall_scale)
    frame_to_world_raw = read_traj_adjusted(traj_fname, args.overall_scale)

    if inds_fname is not None:
        inds_f = open(inds_fname)
        inds = [int(x) for x in inds_f.readline().split()]
        inds = [x - 1 for x in inds]
        if len(inds) != len(frame_to_world_raw):
            print('size of the index array does not correspond to the trajectory')
            exit(1)
        frame_to_world = [None] * len(frame_to_world_GT)
        is_present = [False] * len(frame_to_world_GT)
        for orig_ind, real_ind in enumerate(inds):
            is_present[real_ind] = True
            frame_to_world[real_ind] = frame_to_world_raw[orig_ind]
        for i in range(1, len(frame_to_world)):
            if frame_to_world[i] is None:
                frame_to_world[i] = frame_to_world[i - 1]
        for i in range(len(frame_to_world) - 2, 0, -1):
            if frame_to_world[i] is None:
                frame_to_world[i] = frame_to_world[i + 1]
        del inds
    else:
        frame_to_world = frame_to_world_raw
        is_present = [True] * len(frame_to_world)

    del frame_to_world_raw

    if first is not None:
        frame_to_world_GT = frame_to_world_GT[first:]
        frame_to_world = frame_to_world[first:]
        is_present = is_present[first:]



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


    trans_drift = [[] for i in range(len(range_lengths) - 1)]
    rot_drift   = [[] for i in range(len(range_lengths) - 1)]
    scale_drift = [[] for i in range(len(range_lengths) - 1)]

    print('start measuring')
    total_subseq = (len(frame_to_world) - 1) * (len(frame_to_world) - 2) // 2
    processed = 0
    bar = progressbar.ProgressBar(max_value=total_subseq)
    for fst in range(len(frame_to_world)):
        for lst in range(fst + 1, len(frame_to_world)):
            if not is_present[fst] or not is_present[fst + 1] or not is_present[lst]:
                continue

            fst_to_lst = np.matmul(world_to_frame[lst], frame_to_world[fst])
            fst_to_lst_GT = np.matmul(world_to_frame_GT[lst], frame_to_world_GT[fst])

            fst_to_nxt = np.matmul(world_to_frame[fst + 1], frame_to_world[fst])
            fst_to_nxt_GT = np.matmul(world_to_frame_GT[fst + 1], frame_to_world_GT[fst])

            scale_fix = norm(t(fst_to_nxt_GT)) / norm(t(fst_to_nxt)) \
                        if scale_fix else 1

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

            err_angle = angle(np.matmul(np.transpose(R(fst_to_lst)),
                                                R(fst_to_lst_GT)))
            rot_drift[range_ind].append((180 / np.pi) * err_angle / path_length_GT)
            scale_drift[range_ind].append(100 * abs(path_length / path_length_GT - 1) 
                                                / path_length_GT)
            processed += 1
            bar.update(processed)

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


    errors_list = []
    for range_ind in range(num_ranges):
        for td in trans_drift[range_ind]:
            errors_list.append({'error':td, 'kind':'translation', 'scale':range_ind, 
                                'label':label})
        for rd in rot_drift[range_ind]:
            errors_list.append({'error':rd, 'kind':'rotation', 'scale':range_ind, 
                                'label':label})
        for sd in scale_drift[range_ind]:
            errors_list.append({'error':sd, 'kind':'scale', 'scale':range_ind, 
                                'label':label})
    errors = pd.DataFrame(errors_list)
    return (errors, range_lengths)

errors, range_lengths = get_errors(args.traj, args.inds, args.gt, args.first, 
                                   args.label, args.scale_fix)
if args.other is not None:
    other_errors, _ = get_errors(args.other, args.other_inds, args.gt,
                                 args.first, args.other_label, args.other_scale_fix)
    errors = errors.append(other_errors)

sns.set(style="whitegrid", palette="pastel", color_codes=True)
grid = sns.catplot(data=errors, y='error', row='kind', x='scale',
                   hue='label', kind='violin', split=True, sharey='row', cut=0,
                   inner='quartile', legend=False, 
                   palette=sns.color_palette(palette="Set2", n_colors=2))

title_fontdict = {
    'fontsize': 'xx-large',
    'fontweight' : 'bold',
    'verticalalignment': 'baseline',
    'horizontalalignment': 'center'
}

xlabel_fontdict = {
    'fontsize': 'x-large',
    'fontweight' : 'normal',
    'verticalalignment': 'baseline',
    'horizontalalignment': 'center'
}

#  labels = ['original', 'our']
#  for t, l in zip(grid._legend.texts, labels): t.set_text(l)
grid.axes[0,0].set_title(label_trans, fontdict=title_fontdict)
grid.axes[0,0].set_ylabel('')
grid.axes[1,0].set_title(label_rot, fontdict=title_fontdict)
grid.axes[1,0].set_ylabel('')
grid.axes[2,0].set_title(label_scale, fontdict=title_fontdict)
grid.axes[2,0].set_ylabel('')
grid.axes[2,0].set_xlabel(label_ranges, labelpad=15, fontdict=xlabel_fontdict)
grid.axes[2,0].set_xticklabels([f'[{range_lengths[i]:.1f}, {range_lengths[i+1]:.1f})'
                                for i in range(num_ranges)])

legend = grid.axes[0,0].legend(fontsize='x-large', loc='lower left', bbox_to_anchor=(0.8, 0.85))
#  grid._legend.set_title('')
#  plt.setp(grid._legend.get_texts(), fontsize='large')
#  plt.legend(, title_fontsize='large')

for ax in grid.axes[:,0]:
    ax.tick_params(labelsize='x-large')
    for i, line in enumerate(ax.get_lines()):
        if i % 3 != 1:
            line.set_visible(False)
        else:
            line.set_linestyle('-')
#  ax = grid.axes[0,0]
#  lines = ax.get_lines()
#  l0 = lines[0]
#  l1 = lines[1]
#  l0.set_color('red')
#  l1.set_color('green')
#  rect = mpl.patches.Rectangle((l0.xdata[0], l0.ydata[0]), (l1.xdata[0] - , l1.ydata[0]))
#  collections = ax.findobj(mpl.collections.PolyCollection)
#  c0 = collections[0]

#  print(c0.get_paths()[0])
#  print(l.get_xdata())
#  print(l.get_ydata())

#  fig, (ax_trans, ax_rot, ax_scale) = plt.subplots(3, 1)
#  ax_trans.set_title('translational error distributions (%)')
#  ax_trans.set_xticklabels([])
#  ax_trans.violinplot(trans_drift, showmeans=True)

#  ax_rot.set_title('rotational error distributions (deg/m)')
#  ax_rot.set_xticklabels([])
#  ax_rot.violinplot(rot_drift, showmeans=True)

#  ax_scale.set_title('scale error distributions (%/m)')
#  ax_scale.set_xlabel('ranges of lengths of subsequences (m)')
#  ax_scale.violinplot(scale_drift, showmeans=True)

plt.gcf().subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.07, hspace=0.2)

mng = plt.get_current_fig_manager()
mng.full_screen_toggle()

#  tikzplotlib.save('fig.tex')

plt.show()
