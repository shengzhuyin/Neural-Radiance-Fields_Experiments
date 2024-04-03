#import torch, lightning, np
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from walk_learning import WalkLearner

# load model
model =  torch.load("/scratch/users/akshat7/cv/temp/editnerf/tb_logs/my_model/version_8/checkpoints/walk-nerf-epoch=09.ckpt")
print(model)
