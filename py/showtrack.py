import sys
import numpy as np
import pandas as pd
import quaternion as quat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import norm

def extract_motions(fname):
    tbl = pd.read_csv(fname, sep='\s+', header=None,
                      index_col=0, dtype=np.float64)
    motions = list(map(
        lambda x: (np.array(x[-3:]),
                   quat.as_rotation_matrix(
                       np.quaternion(x[3], x[0], x[1], x[2]))),
        tbl.apply(list, axis=1).tolist()
    ))
    return motions


def draw_motions(axes, motions, color, label):
    centers = np.array([-np.matmul(mot[1].T, mot[0]) for mot in motions])
    dir_vects = np.array([mot[1][2, :] for mot in motions])

    axes.quiver(centers[:, 0], centers[:, 1], centers[:, 2],
              dir_vects[:, 0], dir_vects[:, 1], dir_vects[:, 2], 
                color=color, normalize=True, arrow_length_ratio=0.2, 
                length=0.15, label=label)

def main(argv):
    if (len(argv) != 2):
        print("I need just one argument -- dso output directory")
        return 0

    out_dir = argv[1]
    if (out_dir[-1] == '/'):
        out_dir = out_dir[0:-1]

    actual = extract_motions(out_dir + '/tracked_pos.txt')

    has_predicted = True
    try:
        predicted = extract_motions(out_dir + '/predicted_pos.txt')
    except FileNotFoundError:
        has_predicted = False
        print('no prediction provided')

    has_stereo_matched = True
    try:
        stereo_matched = extract_motions(out_dir + '/matched_pos.txt')
    except FileNotFoundError:
        has_stereo_matched = False
        print('no stereo-matching provided')

    has_ground_truth = True
    try:
        ground_truth = extract_motions(out_dir + '/ground_truth_pos.txt')
    except FileNotFoundError:
        has_ground_truth = False
        print('no ground truth provided')

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.set_yticks([])

    ax.view_init(azim=-90, elev=0)

    xylim = 4
    ax.set_xlim3d(-xylim, xylim)
    ax.set_ylim3d(-xylim, xylim)
    ax.set_zlim3d(0, 2 * xylim)

    draw_motions(ax, actual, 'orange', 'полученая оценка')
    if has_predicted:
        draw_motions(ax, predicted, 'blue')
    if has_ground_truth:
        draw_motions(ax, ground_truth, 'green', 'точная траектория')
    elif has_stereo_matched:
        draw_motions(ax, stereo_matched, 'green')

    ax.scatter3D([0], [0], [0], color='black', label='базовый кадр')

    ax.legend()
    plt.show()

if __name__ == "__main__":
    sys.exit(main(sys.argv))
