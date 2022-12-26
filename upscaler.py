import cv2
import glob
import numpy as np
import os
import torch
import sys
from basicsr.utils import imwrite
from gfpgan import GFPGANer

def main(inputPath, outputPath, upscaleFactor):
    if inputPath.endswith('/'):
        inputPath = inputPath[:-1]
    if os.path.isfile(inputPath):
        img_list = [inputPath]
    else:
        img_list = sorted(glob.glob(os.path.join(inputPath, '*.webp')))

    os.makedirs(outputPath, exist_ok=True)

    scale = float(upscaleFactor)

    if not torch.cuda.is_available():  # CPU
        import warnings
        warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                      'If you really want to use it, please modify the corresponding codes.')
        bg_upsampler = None
    else:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        bg_upsampler = RealESRGANer(
                scale=2,
                model=model,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                device='cuda',
                tile_pad=10,
                pre_pad=0,
                half=True)

    # ------------------------ set up GFPGAN ------------------------
    restorer = GFPGANer(
        upscale=scale,
        model_path=os.path.join('GFPGAN/experiments/pretrained_models', "GFPGANv1.4" + '.pth'), 
        bg_upsampler=bg_upsampler)
        
    if torch.cuda.is_available():
        model.cuda()

    for img_path in img_list:
        basename = os.path.basename(img_path)
        resultPath = os.path.join(outputPath, basename)
        if os.path.exists(resultPath):
            continue
        sys.stdout.write(f'processing: {basename}')
        sys.stdout.flush()
        input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        cropped_faces, restored_faces, restored_img = restorer.enhance(
            input_img,
            paste_back=True,
            weight=0.5)
        imwrite(restored_img, resultPath)

sys.stdout.write(f'started')
sys.stdout.flush()
for line in sys.stdin:
    inputPath, outputPath, upscaleFactor = line.rstrip().split("\0")
    main(inputPath, outputPath, upscaleFactor)
    sys.stdout.write(f'{outputPath}\0ready')
    sys.stdout.flush()