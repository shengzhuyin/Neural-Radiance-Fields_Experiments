import os
import copy

import numpy as np
import torch
import cv2

from run_nerf_helpers import img2mse, mse2psnr

from inputs import config_parser
from dataset import load_data
from model import create_nerf
from rendering import render, render_path
from utils.pidfile import exit_if_job_done, mark_job_done

from torch import optim
import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)

parser = config_parser()
args = parser.parse_args()
torch.set_default_tensor_type('torch.cuda.FloatTensor')
print(f"[INFO] Read args")

# Create log dir and copy the config file
basedir = args.basedir
expname = args.savedir if args.savedir else args.expname
print('Experiment dir:', expname)

EDIT_TYPE = "COLOR"
assert EDIT_TYPE in ["COLOR", "ZOOM"], f"{EDIT_TYPE = }"
print(f"[INFO] Edit type is {EDIT_TYPE}")

NUM_IMAGES = -1
if __name__ != '__main__':
    NUM_IMAGES = 2
if NUM_IMAGES !=-1:
    print(f"[WARNING] Loading only {NUM_IMAGES = } images. This is OK if you are using the code for interpolating between styles.")

images, poses, style, i_test, i_train, bds_dict, dataset, hwfs, near_fars, style_inds = load_data(args, num_images = NUM_IMAGES)
_, poses_test, style_test, hwfs_test, nf_test = images[i_test], poses[i_test], style[i_test], hwfs[i_test], near_fars[i_test]
_, poses_train, style_train, hwfs_train, nf_train = images[i_train], poses[i_train], style[i_train], hwfs[i_train], near_fars[i_train]
NUM_IMAGES = images.shape[0]
print(f"[INFO] Loaded data")
print(f"[INFO] {NUM_IMAGES = }")

os.makedirs(os.path.join(basedir, expname), exist_ok=True)

# np.save(os.path.join(basedir, expname, 'poses.npy'), poses_train.cpu())
# np.save(os.path.join(basedir, expname, 'hwfs.npy'), hwfs_train.cpu())
f = os.path.join(basedir, expname, 'args.txt')
with open(f, 'w') as file:
    for arg in sorted(vars(args)):
        attr = getattr(args, arg)
        file.write('{} = {}\n'.format(arg, attr))
if args.config is not None:
    f = os.path.join(basedir, expname, 'config.txt')
    with open(f, 'w') as file:
        file.write(open(args.config, 'r').read())

class WalkLearner(pl.LightningModule):
    def __init__(self, args, dataset, nf_train, nf_test, poses_train, poses_test, hwfs_train, hwfs_test, style_train, style_test):
        super().__init__()
        self.args = args
        self.dataset = dataset
        self.nf_train = nf_train
        self.nf_test = nf_test
        self.poses_train = poses_train
        self.poses_test = poses_test
        self.hwfs_train = hwfs_train
        self.hwfs_test = hwfs_test
        self.style_train = style_train
        self.style_test = style_test
        self.render_kwargs_train, self.render_kwargs_test, self.start, self.grad_vars, self.optimizer = create_nerf(self.args)
        self.model = self.render_kwargs_train['network_fn']
        self.model_fine = self.render_kwargs_train['network_fine']
        for param in self.grad_vars:
            param.requires_grad = False
        
        if EDIT_TYPE == "COLOR":
            self.w = torch.nn.Parameter(torch.randn((args.style_dim//2, 3), requires_grad = True))
        elif EDIT_TYPE == "ZOOM":
            self.w = torch.nn.Parameter(torch.randn((args.style_dim, 1), requires_grad = True)) # learned walk
        
        # shift models to self.w.device
        self.model.to(self.w.device)
        self.model_fine.to(self.w.device)
        
    def configure_optimizers(self):
        return optim.Adam([self.w], lr = 1e-5)
        
    def lies_in(self, alpha, min_, max_):
        if alpha >= min_ and alpha <= max_:
            return True
        return False
        
    def check_alpha(self, alpha):
        # for i in range(alpha.shape[0]):
        #     assert self.lies_in(alpha[i][0], 0, 1), "alpha[0] = {}".format(alpha[0])
        #     assert self.lies_in(alpha[i][1], 0, 1), "alpha[1] = {}".format(alpha[1])
        #     assert self.lies_in(alpha[i][2], 0, 1), "alpha[2] = {}".format(alpha[2])
        assert self.lies_in(alpha[0], 0, 1), "alpha[0] = {}".format(alpha[0])
        assert self.lies_in(alpha[1], 0, 1), "alpha[1] = {}".format(alpha[1])
        assert self.lies_in(alpha[2], 0, 1), "alpha[2] = {}".format(alpha[2])
    
    def perform_edit(self, original_image, alpha, type):
        assert type in ["COLOR", "ZOOM"], f"{type = }"
        if type == "COLOR":
            assert alpha.shape[0] == 3, f"{alpha.shape = }"
            assert len(original_image.shape) == 2 and original_image.shape[1] == 3, f"{original_image.shape = }"
            self.check_alpha(alpha)
            
            # convert (HxW, 3) to (H, W, 3)
            img_size = int(np.sqrt(original_image.shape[0]))
            image = original_image.view(img_size, img_size, 3)
            max_, _ = torch.max(image, dim = -1, keepdim = True)
            image = image / (max_ + 1e-5)
            
            # do the edit by balancing the color channels
            new_image = torch.zeros_like(image)
            new_image = torch.matmul(image, torch.diag(alpha))
            max_, _ = torch.max(new_image, dim = -1, keepdim = True)
            new_image = new_image / (max_ + 1e-5) # normalize
            
            mask = ((torch.min(image, dim=-1)[0]) >= 1-(1e-2))  # once again, numerical accuracy is the bane of my existence
            mask = mask.unsqueeze(-1).expand(-1, -1, 3)
            new_image[mask] = 1.0
            
            new_image = new_image.view(original_image.shape)
            assert new_image.shape == original_image.shape, f"{new_image.shape = }, {original_image.shape = }"
            return new_image
        
        elif type == "ZOOM":
            scale_factor = torch.exp(alpha)    # to allow for positive and negative zoom
            assert alpha.shape[0] == 1, f"{alpha.shape = }"
            assert len(original_image.shape) == 2 and original_image.shape[1] == 3, f"{original_image.shape = }"
            
            img_size = int(np.sqrt(original_image.shape[0]))
            image = original_image.view(img_size, img_size, 3)
            
            new_height = int(img_size * scale_factor)
            new_width = int(img_size * scale_factor)
            np_image = image.detach().cpu().numpy()
            new_image = cv2.resize(np_image, (new_width, new_height), interpolation = cv2.INTER_LINEAR)
            
            if scale_factor > 1:
                crop_start_x = (new_width - img_size) // 2
                crop_start_y = (new_height - img_size) // 2
                crop_end_x = crop_start_x + img_size
                crop_end_y = crop_start_y + img_size
                new_image = new_image[crop_start_y:crop_end_y, crop_start_x:crop_end_x]
            
            elif scale_factor < 1:
                pad_height = max(0, img_size - new_height)
                pad_width = max(0, img_size - new_width)

                # Calculate the padding on each side
                top_pad = pad_height // 2
                left_pad = pad_width // 2
                bottom_pad = pad_height - top_pad
                right_pad = pad_width - left_pad
                
                new_image = cv2.copyMakeBorder(
                    new_image,
                    top_pad,
                    bottom_pad,
                    left_pad,
                    right_pad,
                    cv2.BORDER_CONSTANT,
                    value=[1, 1, 1]  # You can specify the padding color here (black in this case)
                )
            
            new_image = torch.tensor(new_image).to(image.device)
            max_, _ = torch.max(new_image, dim = -1, keepdim = True)
            new_image = new_image / (max_ + 1e-5) # normalize
            
            new_image = new_image.view(original_image.shape)
            assert new_image.shape == original_image.shape, f"{new_image.shape = }, {original_image.shape = }"
            return new_image
        
    def interpol(self, z, alpha, w):
        if EDIT_TYPE == "COLOR":
            assert z.shape[0] == args.style_dim//2, "z.shape = {}".format(z.shape)
            assert alpha.shape[0] == 3, "alpha.shape = {}".format(alpha.shape)
            assert w.shape == (args.style_dim//2, 3), "w.shape = {}".format(w.shape)
        elif EDIT_TYPE == "ZOOM":
            assert alpha.shape[0] == 1, "alpha.shape = {}".format(alpha.shape)
            assert w.shape == (args.style_dim, 1), "w.shape = {}".format(w.shape)
            
        return z + torch.matmul(w, alpha)
    
    def interpol_style(self, style, alpha, w):
        assert style.shape[0] == args.style_dim, "style.shape = {}".format(style.shape)
        if EDIT_TYPE == "COLOR":
            assert alpha.shape[0] == 3, "alpha.shape = {}".format(alpha.shape)
            assert w.shape == (args.style_dim//2, 3), "w.shape = {}".format(w.shape)
            # get shape style
            shape_style = style[:args.style_dim//2]
            color_style = style[args.style_dim//2:]
            new_color_style = self.interpol(color_style, alpha, w)
            
            # join to make final style
            return torch.cat((shape_style, new_color_style), dim = 0)
        
        elif EDIT_TYPE == "ZOOM":
            assert alpha.shape[0] == 1, "alpha.shape = {}".format(alpha.shape)
            assert w.shape == (args.style_dim, 1), "w.shape = {}".format(w.shape)
            
            new_style = self.interpol(style, alpha, w)
            return new_style
    
    def nerf_forward(self, H, W, focal, style, chunk, rays, viewdirs_reg):
        '''
        produce rgb map and other outputs for input style z
        '''
        rgb, disp, acc, extras = render(H, W, focal, style=style, chunk=chunk, rays=rays, viewdirs_reg=viewdirs_reg, **self.render_kwargs_train)
        return rgb, disp, acc, extras
    
    def training_step(self, batch, batch_idx):
        # forget the batch, sample from self.dataset
        batch_rays, target_s, style, H, W, focal, near, far, viewdirs_reg = self.dataset.get_data_batch(train_fn=self.render_kwargs_train)
        
        # check style
        assert torch.norm(style - style.mean(dim = 0)) < 1e-2, f"all styles are not the same {style = }, {style - style.mean(dim = 0)}, {torch.norm(style - style.mean(dim = 0)) = }"
        # print(f"[INFO] All styles are same.")
        style_num = style.shape[0]
        style = style.mean(dim = 0)
        
        if EDIT_TYPE == "COLOR":
            alpha = torch.rand(3)
            alpha = alpha / torch.norm(alpha)   # normalize to lie on unit sphere
        elif EDIT_TYPE == "ZOOM":
            alpha = 2*torch.rand(1)-1  # in [-1,1)
        
        # alpha = torch.tensor([0.0, 0.0, 1.0])
        # print(f"[WARNING] polar {alpha = }")
        
        # print(f"[DEBUG] style.shape = {style.shape}")
        # print(f"[DEBUG] alpha.shape = {alpha.shape}")
        interpolated_style = self.interpol_style(style, alpha, self.w)
        assert interpolated_style.shape == style.shape, f"{interpolated_style.shape = }, {style.shape = }"
        
        rgb, disp, acc, extras = self.nerf_forward(H, W, focal, torch.stack([style] * style_num), chunk = self.args.chunk, rays = batch_rays, viewdirs_reg = viewdirs_reg)
        # print(f"[INFO] rgb.shape = {rgb.shape}")
        interpol_rgb, interpol_disp, interpol_acc, interpol_extras = self.nerf_forward(H, W, focal, torch.stack([interpolated_style] * style_num), chunk = self.args.chunk, rays = batch_rays, viewdirs_reg = viewdirs_reg)
        # print(f"[INFO] interpol_rgb.shape = {interpol_rgb.shape}")
        
        edited_image = self.perform_edit(rgb, alpha, type = EDIT_TYPE)

        # check for rgb
        assert torch.isnan(rgb).sum() == 0, f"{torch.isnan(rgb).sum() = }"
        assert torch.isnan(interpol_rgb).sum() == 0, f"{torch.isnan(interpol_rgb).sum() = }"
        assert torch.isnan(edited_image).sum() == 0, f"{torch.isnan(edited_image).sum() = }"
        
        loss = img2mse(interpol_rgb, edited_image)
        
        self.log("loss", loss)
        self.log("w_norm", self.w.norm())
        return loss

print(f"[INFO] Creating model ...")
pl_model = WalkLearner(args, dataset, nf_train, nf_test, poses_train, poses_test, hwfs_train, hwfs_test, style_train, style_test)

# print(f"[INFO] Loading model from existing checkpoint ...")
# pl_model = WalkLearner.load_from_checkpoint("/scratch/users/akshat7/cv/temp/editnerf/tb_logs/my_model/version_18/checkpoints/last.ckpt", args = args, dataset = dataset, nf_train = nf_train, nf_test = nf_test, poses_train = poses_train, poses_test = poses_test, hwfs_train = hwfs_train, hwfs_test = hwfs_test, style_train = style_train, style_test = style_test)

# make dummy torch dataset 
from torch.utils.data import Dataset, DataLoader
class DummyDataset(Dataset):
    def __init__(self, NUM_IMAGES):
        self.len = NUM_IMAGES
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = torch.randn(size = (NUM_IMAGES, 3, 1)).to(device)
    def __getitem__(self, idx):
        return self.data[idx]
    def __len__(self):
        return self.len

dataset = DummyDataset(NUM_IMAGES)
dataloader = DataLoader(dataset, batch_size = 32, generator=torch.Generator(device='cuda'))

# make model saving callback
from pytorch_lightning.callbacks import ModelCheckpoint
checkpoint_callback = ModelCheckpoint(
    monitor='loss',
    verbose=True,
    save_last = True,
    every_n_epochs = int(5e2),
    filename='walk-nerf-{epoch:02d}',
)

def run():
    # make pl trainer
    tb_logger = TensorBoardLogger("tb_logs", name="my_model")
    trainer = pl.Trainer(accelerator = "gpu", devices = 4, max_epochs = -1, logger = tb_logger, log_every_n_steps = 1, enable_checkpointing=True, callbacks=[checkpoint_callback])
    trainer.fit(pl_model, dataloader)

if __name__ == '__main__':
    run()
