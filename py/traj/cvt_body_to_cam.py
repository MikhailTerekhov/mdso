from traj import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('traj', help='File with the body trajectory')
parser.add_argument('cam_to_body', help='File with the camera to '
                    'body transformation')
parser.add_argument('-o', '--output', default='cam_to_world.txt',
                    help='Result filename')
parser.add_argument('-i', '--inverse', action='store_true',
                    help='Do the inverse conversion')
args = parser.parse_args()

body_to_world = extract_motions(args.traj)
cam_to_body = to_motion(open(args.cam_to_body, 'r').readline())
if args.inverse:
    cam_to_body = cam_to_body.inverse()
cam_to_world = [btw * cam_to_body for btw in body_to_world]

out = open(args.output, 'w')
for ctw in cam_to_world:
    print(ctw, file=out)
