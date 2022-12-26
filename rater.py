import argparse
import gc

import open_clip
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from PIL import Image

import urllib
from io import BytesIO

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default="sac+ava+logos-h14-linearMSE.pth", help="path to the model")
parser.add_argument(
    "--device", type=str, default="cuda", help="device to use", choices=["cuda", "cpu"]
)
args = parser.parse_args()

class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol="emb", ycol="avg_rating"):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            # nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)
    
model = MLP(1024)
s = torch.load(args.model_path, map_location=torch.device(args.device))
model.load_state_dict(s)
model.to(args.device)
model.eval()
model2, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-H-14", pretrained="laion2b_s32b_b79k", device=args.device
)

sys.stdout.write(f'started')
sys.stdout.flush()

for line in sys.stdin:
    url = line.rstrip()
    u = urllib.request.urlopen(url)
    raw_data = u.read()
    u.close()
    pil_image = Image.open(BytesIO(raw_data)).convert("RGB")
    image = preprocess(pil_image).unsqueeze(0).to(args.device)

    with torch.no_grad():
        image_features = model2.encode_image(image)

    im_emb_arr = normalized(image_features.cpu().detach().numpy() )

    prediction = model(
        torch.from_numpy(im_emb_arr)
        .to(args.device)
        .type(torch.cuda.FloatTensor if args.device == "cuda" else torch.FloatTensor)
    )

    sys.stdout.write(f'{url} {prediction.item()}')
    sys.stdout.flush()