from traj import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('traj', help='File with the trajectory')
parser.add_argument('output', help='File with the scaled trajectory')
parser.add_argument('scale', help='scale factor', type=float)
args = parser.parse_args()

def scale(motion, scale_factor):
    result = motion
    result.t *= scale_factor
    return result

traj = extract_motions(args.traj)
scaled = [scale(m, args.scale) for m in traj]
out = open(args.output, 'w')
for m in scaled:
    print(m, file=out)
