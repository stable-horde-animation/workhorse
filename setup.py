import gdown
import os
import urllib.request
import subprocess
from utils import run

def setup():
    run(['pip', 'install', '-r requirements.txt'])
    if not os.path.exists("workhorse/pretrained_models"):
        os.mkdir("workhorse/pretrained_models")

    gdown.download_folder(url='https://drive.google.com/drive/u/1/folders/1q8110-qp225asX3DQvZnfLfJPkCHmDpy', output='workhorse/pretrained_models')

    urllib.request.urlretrieve('https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth', 'workhorse/GFPGAN/experiments/pretrained_models/GFPGANv1.4.pth')

setup()