import sys
import os
import argparse
import numpy as np
import OpenEXR
import Imath
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("frames_dir", help="A directory with frames to be "
                    "converted")
parser.add_argument("img_dir", help="A directory to store .jpg images")
parser.add_argument("depths_dir", help="A directory to store .bin depth images")
args = parser.parse_args()

def gamma_encode(channel):
    channel = np.clip(channel, 0, 1)
    return np.where(channel <= 0.0031308,
                    12.92 * channel,
                    1.055 * np.power(channel, 1 / 2.4) - 0.055)

def convert(fname):
    print(fname)
    img = OpenEXR.InputFile(fname)
    header = img.header()
    dw = header['dataWindow']
    img_size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    r, g, b, a, d = [np.fromstring(c, dtype=np.float32).reshape(img_size)
                     for c in img.channels('RGBAZ', pt)]
    r, g, b = [gamma_encode(c) for c in (r, g, b)]
    img = np.dstack((b, g, r))
    img = (img * 255.999).astype(np.uint8)
    return img, d

if __name__ == '__main__':
    frames_dir = Path(args.frames_dir)
    img_dir = Path(args.img_dir)
    depths_dir = Path(args.depths_dir)
    assert frames_dir.exists()
    assert img_dir.exists()
    assert depths_dir.exists()
    fnames = sorted(frames_dir.iterdir())
    for fname in fnames:
        if fname.suffix == '.exr':
            print(fname)
            img, d = convert(str(fname))
            img_path = str(img_dir.joinpath(fname.stem + '.jpg'))
            depths_path = str(depths_dir.joinpath(fname.stem + '.bin'))
            print(f'converting {fname} -> {img_path} ; {depths_path}')
            cv2.imwrite(img_path, img)
            d.tofile(depths_path)
