import random

import torch
import os
import numpy as np

from rendering import render_path
from dataset import load_data
from inputs import config_parser
from model import create_nerf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(42)

def interpolate():
    INTEROP_TYPE = "COLOR_TEST"
    INTEROP_TYPE_LIST = ["COLOR", "SHAPE", "COLOR_TEST", "ZOOM_TEST"]
    assert INTEROP_TYPE in INTEROP_TYPE_LIST, f"interop type should be one of {INTEROP_TYPE_LIST}, passed {INTEROP_TYPE}"
    
    parser = config_parser()
    args = parser.parse_args()

    images, poses, style, i_test, i_train, bds_dict, dataset, hwfs, near_fars, _ = load_data(args, num_images = 4)
    images_test, poses_test, style_test, hwfs_test, nf_test = images[i_test], poses[i_test], style[i_test], hwfs[i_test], near_fars[i_test]
    images_train, poses_train, style_train, hwfs_train, nf_train = images[i_train], poses[i_train], style[i_train], hwfs[i_train], near_fars[i_train]
    # print(f"[DEBUG] {i_test = }, {style_test.shape = }, {style_test = }")
    # print(f"[DEBUG] {style.shape = }, {style = }")

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)

    np.save(os.path.join(basedir, expname, 'poses.npy'), poses_train.cpu())
    np.save(os.path.join(basedir, expname, 'hwfs.npy'), hwfs_train.cpu())

    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)
    
    # print(f"[DEBUG] {poses_test.shape = }, {style_test.shape = }, {hwfs_test.shape = }, {nf_test.shape = }")
    
    # pick one pose
    n = poses_test.shape[0]
    # index = random.randint(0, n - 1)
    # color_idx = random.randint(0, n - 1)
    index, color_idx = 1, 0
    print(f"[DEBUG] {n = }, {index = }, {color_idx = }")
    N = 128
    poses_test = torch.stack([poses_test[index]] * N, dim = 0)
    hwfs_test = torch.stack([hwfs_test[index]] * N, dim = 0)
    nf_test = torch.stack([nf_test[index]] * N, dim = 0)
    
    def interop(style, second_style, t):
        assert t>=0 and t<=1
        if t>0.5:
            t = 1-t
        t = t*2
        return torch.lerp(input = style, end = second_style, weight = t)
    
    def interpolate_color(base_style, second_style, n):
        assert (base_style - second_style).norm() > 1e-5, f"Styles are equal"
        style_sigma = base_style[:args.style_dim//2].clone() # shape is fixed
        base_style_color = base_style[args.style_dim//2:].clone()
        base_second_style = second_style[args.style_dim//2:].clone()
        color_interpolations = [interop(base_style_color, base_second_style, i/n) for i in range(n)]
        color_interpolations = [torch.cat([style_sigma, i], dim = 0) for i in color_interpolations]
        color_interpolations = torch.stack(color_interpolations)
        return color_interpolations

    def interpolate_shape(base_style, second_style, n):
        assert (base_style - second_style).norm() > 1e-5, f"Styles are equal"
        style_color = base_style[args.style_dim//2:].clone() # color is fixed
        base_style_sigma = base_style[:args.style_dim//2].clone()
        base_second_style = second_style[:args.style_dim//2].clone()
        shape_interpolations = [interop(base_style_sigma, base_second_style, i/n) for i in range(n)]
        shape_interpolations = [torch.cat([i, style_color], dim = 0) for i in shape_interpolations]
        shape_interpolations = torch.stack(shape_interpolations)
        return shape_interpolations
    
    def load_w_fromcheckpoint():
        import torch, pytorch_lightning as pl, numpy as np
        from walk_learning import WalkLearner
        # model =  torch.load("/scratch/users/akshat7/cv/temp/editnerf/tb_logs/my_model/version_17/checkpoints/last.ckpt")
        model =  torch.load("/scratch/users/akshat7/cv/temp/editnerf/tb_logs/my_model/version_35/checkpoints/last.ckpt")    # for zoom edits
        w = model["state_dict"]['w']
        return w
    
    def interpolate_vector(base_style, w, walk_direction:int, n):
        assert walk_direction >=-1 and walk_direction <= 2, f"walk_direction should be -1, 0, 1 or 2, passed {walk_direction}"
        
        if w.shape[1] == 3:
            # this is a color walk, not a zoom walk
            assert INTEROP_TYPE == "COLOR_TEST", f"interop type should be COLOR_TEST, passed {INTEROP_TYPE}"
            print(f"[INFO] Doing a color walk ...")
            assert w.shape == (args.style_dim//2, 3), f"{w.shape = }"
            
            #get alphas evenly spaced from 0 to 2
            alphas = torch.linspace(0, 2, n)
            print(f"[DEBUG] Generated {alphas = }")
            
            if walk_direction != -1:    # interpolate in a single color
                raise NotImplementedError(f"walk_direction = {walk_direction} not implemented")
            
                # alpha_ = torch.zeros(size = (n, 3))
                # alpha_[:, walk_direction] = alphas
                # print(f"[DEBUG] Generated {alpha_ = }")
                
                # displacement = torch.matmul(alpha_, w.T)
                # new_ = torch.zeros(size = (n, args.style_dim))
                # for i in range(n):
                #     disp = torch.cat([torch.zeros(args.style_dim//2), displacement[i]], dim = 0)
                #     assert disp.shape == (args.style_dim, ), f"{disp.shape = }"
                #     new_[i] = disp
                # displacement = new_
                
                # assert displacement.shape == (n, args.style_dim), f"{displacement.shape = }"
                
            else: # interpolate in all colors
                # get alpha
                np.random.seed(42)
                torch.manual_seed(42)
                p1, p2 = torch.tensor([1, 0, 0], dtype = torch.float32), torch.tensor([0, 1, 0], dtype = torch.float32)
                p1, p2 = p1 / p1.norm(), p2 / p2.norm()
                
                theta_0, theta_1 = torch.acos(p1[2]), torch.acos(p2[2])
                phi_0, phi_1 = torch.atan2(p1[1], p1[0]), torch.atan2(p2[1], p2[0])
                t = torch.linspace(0, 2, n)
                t[t > 1] = 2 - t[t > 1]
                
                # interpolate theta and phi
                thetas = theta_0 * (1 - t) + theta_1 * t
                phis = phi_0 * (1 - t) + phi_1 * t
                
                # convert back to alpha
                alpha_ = torch.stack([torch.sin(thetas) * torch.cos(phis), torch.sin(thetas) * torch.sin(phis), torch.cos(thetas)], dim = 1)
                assert alpha_.shape == (n, 3), f"{alphas.shape = }"
                print(f"[DEBUG] Generated {alpha_ = }")
                
                displacement = torch.matmul(alpha_, w.T)
                new_ = torch.zeros(size = (n, args.style_dim))
                for i in range(n):
                    disp = torch.cat([torch.zeros(args.style_dim//2), displacement[i]], dim = 0)
                    assert disp.shape == (args.style_dim, ), f"{disp.shape = }"
                    new_[i] = disp
                displacement = new_
                
                assert displacement.shape == (n, args.style_dim), f"{displacement.shape = }"
                
            print(f"[DEBUG] {displacement.norm() = }, {displacement}")
            
            interpolated_styles = base_style.reshape(1,-1) + displacement
            assert interpolated_styles.shape == (n, args.style_dim), f"{interpolated_styles.shape = }"
            
            return interpolated_styles

        elif w.shape[1] == 1:
            # this is a zoom walk, not a color walk
            assert INTEROP_TYPE == "ZOOM_TEST", f"interop type should be ZOOM_TEST, passed {INTEROP_TYPE}"
            print(f"[INFO] Doing a zoom walk ...")
            assert w.shape == (args.style_dim, 1), f"{w.shape = }"
            
            #get alphas evenly spaced from 0 to 2
            alphas = torch.linspace(0, 2, n)
            print(f"[DEBUG] Generated {alphas = }")
            
            displacements = torch.matmul(alphas.reshape(-1, 1), w.T)
            assert displacements.shape == (n, args.style_dim), f"{displacements.shape = }"
            
            interpolated_styles = base_style.reshape(1,-1) + displacements
            print(f"[DEBUG] {displacements.norm() = }, {displacements}")
            print(f"[DEBUG] {base_style.shape = }, {base_style = }")
            print(f"[DEBUG] {interpolated_styles.shape = }, {interpolated_styles = }")
            
            return interpolated_styles
        
        else:
            raise RuntimeError(f"walk vector should be of shape (N, 3) or (N, 1), passed {w.shape = }")
    
    base_style = style_test[index]
    second_style = style_test[color_idx]
    assert base_style.shape == (args.style_dim, )

    if INTEROP_TYPE == "COLOR":
        style_test = interpolate_color(base_style, second_style, N)
        print(f"[DEBUG] {style_test[0] - style_test[1] = }")
    elif INTEROP_TYPE == "SHAPE":
        style_test = interpolate_shape(base_style, second_style, N)
        print(f"[DEBUG] {style_test[0] - style_test[1] = }")
    else : 
        w = load_w_fromcheckpoint()
        print(f"[INFO] Loaded walk vector {w = }")
        style_test = interpolate_vector(base_style, w, walk_direction = -1, n = N)
        print(f"[DEBUG] {style_test[0] - style_test[1] = }")
    
    print(f"[DEBUG] {poses_test.shape = }, {style_test.shape = }, {hwfs_test.shape = }, {nf_test.shape = }")
        
    with torch.no_grad():
        # pick one poses_test, hwfs_test, nf_test
        # and interpolate between two styles (style_test), keep the shape same and interpolate colors.
        
        path = f"interpolation_INTERPO_{INTEROP_TYPE}"
        testsavedir = os.path.join(basedir, expname, path)
        os.makedirs(testsavedir, exist_ok=True)
        _, _, psnr = render_path(poses_test.to(device), style_test, hwfs_test, args.chunk, render_kwargs_test, nfs=nf_test, gt_imgs=None, savedir=testsavedir)
        print('Saved interpolation set w/ psnr', psnr)

if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    interpolate()
