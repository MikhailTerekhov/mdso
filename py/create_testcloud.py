import numpy as np
import argparse
from pathlib import Path
from traj.traj import *
from bisect import bisect_left
import progressbar
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('dir', help='A directory with dso output')
parser.add_argument('out', help='Output file name')
parser.add_argument('-l', '--segment_len', type=int, nargs='?', default=100,
                    help='Size of the segment that will be aligned')
parser.add_argument('-s', '--scale', action='store_true',
                    help='Do we need to align the scale of segments?')
parser.add_argument('-d', '--stddev', type=float, nargs='?', default=2.5,
                    help='Stddev threshold')


args = parser.parse_args()

def read_cloud(fname):
    lines = open(fname).readlines()
    lines = lines[11:]
    data = np.array([[float(x) for x in line.split()] for line in lines])
    points  = data[:,:3]
    colors  = data[:,3:6]
    stddevs = data[:,6]
    stddevs.reshape(stddevs.size)
    return points, colors, stddevs

def write_cloud(points, colors, fname):
    f = open(fname, 'w')
    print(f'''ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header''', file=f)
    for p, c in zip(points, colors):
        c = [int(x) for x in c]
        print(*p, *c, file=f)


outDir = Path(args.dir)
assert outDir.exists()

gt_traj = extract_motions(outDir.joinpath('frame_to_world_GT.txt'))
vo_traj = extract_motions(outDir.joinpath('tracked_frame_to_world.txt'))
aligned_traj = []
for i in range(0, len(vo_traj), args.segment_len):
    chunk_len = min(len(vo_traj) - i, args.segment_len)
    gt_chunk = gt_traj[i:i+chunk_len]
    vo_chunk = vo_traj[i:i+chunk_len]
    aligned_chunk = align(vo_chunk, gt_chunk, args.scale)
    aligned_traj += aligned_chunk

timestamps = np.loadtxt(outDir.joinpath('timestamps.txt'), dtype=np.int64)

aligned_f = open('aligned.txt', 'w')
for m in aligned_traj:
    print(m, file=aligned_f)


res_points = []
res_colors = []
total_stddevs = []
fnames = list(outDir.iterdir())
fnames.sort()
for fname in progressbar.progressbar(fnames):
    stem = str(fname.stem)
    if stem[:2] == 'kf':
        ts = int(stem[2:])
        ind = bisect_left(timestamps, ts) 
        assert timestamps[ind] == ts

        points, colors, stddevs = read_cloud(fname)
        total_stddevs += list(stddevs)
        for pi in range(points.shape[0]):
            if stddevs[pi] > args.stddev:
                continue
            p = points[pi,:].reshape(3)
            p = aligned_traj[ind] * vo_traj[ind].inverse() * p
            c = colors[pi,:]
            res_points.append(p)
            res_colors.append(c.reshape(3))

write_cloud(res_points, res_colors, args.out)

total_stddevs.sort()
plt.plot(total_stddevs, np.linspace(0, 1, len(total_stddevs)))
plt.show()
