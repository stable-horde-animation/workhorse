import gdown
import os
import urllib.request
from utils import run

def setup():
    run(['pip', 'install', '-r requirements.txt'])
    run(['git', 'clone', 'https://github.com/google-research/frame-interpolation', 'frame_interpolation'])
    if not os.path.exists("pretrained_models"):
        os.mkdir("pretrained_models")

    gdown.download_folder(url='https://drive.google.com/drive/u/1/folders/1q8110-qp225asX3DQvZnfLfJPkCHmDpy', output='pretrained_models')

    run(['git', 'clone', 'https://github.com/TencentARC/GFPGAN.git'])
    run(['python', 'setup.py', 'develop'], 'GFPGAN')
    urllib.request.urlretrieve('https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth', 'GFPGAN/experiments/pretrained_models/GFPGANv1.4.pth')

setup()