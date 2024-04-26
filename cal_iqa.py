import cv2
import glob
import os
import numpy as np
from os import path as osp
from utils import utils_image as util
from rich.progress import track
from natsort import natsorted
import pyiqa
import torch


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
psnr_metric = pyiqa.create_metric('psnr').to(device)
ssim_metric = pyiqa.create_metric('ssim').to(device)
lpips_metric = pyiqa.create_metric('lpips', device=device)


path = 'path_to_results'
gt_path = './test_video/clean/'

folders = os.listdir(path)
folders = natsorted(folders)

psnr = []
ssim = []
lpips = []

for folder in folders:
    # print(folder)
    imgs = natsorted(glob.glob(osp.join(path, folder, '*.png')))
    imgs_gt = natsorted(glob.glob(osp.join(gt_path, folder, '*.jpg')))
    psnr_folder = []
    ssim_folder = []
    lpips_folder = []
    
    for i in track(range(len(imgs))):
        psnr_folder.append(psnr_metric(imgs[i], imgs_gt[i]))
        ssim_folder.append(ssim_metric(imgs[i], imgs_gt[i]))
        lpips_folder.append(lpips_metric(imgs[i], imgs_gt[i]))

    psnr.append(np.mean(psnr_folder))
    ssim.append(np.mean(ssim_folder))
    lpips.append(np.mean(lpips_folder))

    print(folder)
    print(f'psnr: {np.mean(psnr_folder):.2f}, ssim: {np.mean(ssim_folder):.4f}, lpips: {np.mean(lpips_folder):.4f}')

print('psnr: ', np.mean(psnr))
print('ssim: ', np.mean(ssim))
print('lpips: ', np.mean(lpips))
    
    