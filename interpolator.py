import argparse
from io import BytesIO
from math import floor

import numpy as np
from typing import Iterable, List

from tqdm import tqdm
import os
import sys

import tensorflow as tf
import tensorflow_io as tfio
from typing import Generator, Iterable, List, Optional

from frame_interpolation.eval import util, interpolator as interpolate
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default="sac+ava+logos-h14-linearMSE.pth", help="path to the model")
# parser.add_argument("--lowVram", type=, default="sac+ava+logos-h14-linearMSE.pth", help="path to the model")
parser.add_argument("--lowVram", action='store_true', help="use if low vram")
# parser.add_argument(
#     "--device", type=str, default="cuda", help="device to use", choices=["cuda", "cpu"]
# )
args = parser.parse_args()
shape = None
if(args.lowVram):
    shape = [2,2]
# TODO: TBD if should allow CPU interpolation
interpolator = interpolate.Interpolator("pretrained_models/film_net/Style/saved_model", 64, shape)
    
_UINT8_MAX_F = float(np.iinfo(np.uint8).max)

def interpolate_recursively_from_files(
    frames: List[str], times_to_interpolate: int,
    interpolator: interpolate.Interpolator, start: int) -> Iterable[np.ndarray]:
  n = len(frames)
  start_input = floor(start / (2**(times_to_interpolate) - 1))
#   bar = tqdm(total=num_frames, ncols=100, colour='green')
  for i in range(start_input + 1, n):
    sys.stdout.write(f'processing: {i}/{n}')
    sys.stdout.flush()
    yield from util._recursive_generator(
        read_image(frames[i - 1]), read_image(frames[i]), times_to_interpolate,
        interpolator)
  # Separately yield the final frame.
  yield read_image(frames[-1])

def read_image(filename: str) -> np.ndarray:
    image_data = tf.io.read_file(filename)
    image = rgba2rgb(tfio.image.decode_webp(image_data))
    image_numpy = tf.cast(image, dtype=tf.float32).numpy()
    return image_numpy / _UINT8_MAX_F

def rgba2rgb(rgba, background=(255,255,255)):
    row, col, ch = tf.shape(rgba)
    if ch == 3:
        return rgba
    assert ch == 4, 'RGBA image has 4 channels.'
    r, g, b, a = tf.unstack(tf.cast(rgba, tf.float32), axis=-1)
    a =  tf.cast(a, tf.float32) / 255.0
    R, G, B = background
    r = r * a + (1.0 - a) * R
    g = g * a + (1.0 - a) * G
    b = b * a + (1.0 - a) * B
    rgb = tf.stack([r,g,b], axis=-1)
    return tf.cast(rgb, tf.uint8)

def write_image(filename: str, image: np.ndarray) -> None:
    """Writes a float32 3-channel RGB ndarray image, with colors in range [0..1].
    Args:
        filename: The output filename to save.
        image: A float32 3-channel (RGB) ndarray with colors in the [0..1] range.
    """
    image_in_uint8_range = np.clip(image * _UINT8_MAX_F, 0.0, _UINT8_MAX_F)
    image_in_uint8 = (image_in_uint8_range + 0.5).astype(np.uint8)

    image_data = tf.io.encode_png(image_in_uint8)
    image_bytes = image_data.numpy()
    image = Image.open(BytesIO(image_bytes))
    image.save(filename, 'webp')

sys.stdout.write(f'started')
sys.stdout.flush()

for line in sys.stdin:
    inputPath, outputPath, iterations = line.rstrip().split("\0")

    files = tf.io.gfile.glob(f'{inputPath}/*.webp')
    
    start = 0
    if os.path.isdir(outputPath):
        old_frames = tf.io.gfile.glob(f'{outputPath}/*.webp')
        for old_frame in old_frames:
            basename = os.path.basename(old_frame)
            idxstr, ext = basename.split('.')
            idx = int(idxstr) + 1
            if(idx > start):
                start = idx
    else:
        tf.io.gfile.makedirs(outputPath)

    frames = interpolate_recursively_from_files(
            files, int(iterations), interpolator, start)
            
    for idx, frame in enumerate(frames):
        write_image(f'{outputPath}/{(start + idx):06d}.webp', frame)
        
    sys.stdout.write(f'{outputPath}\0ready')
    sys.stdout.flush()