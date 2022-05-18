import os
from copy import deepcopy

import torch
import cv2
import numpy as np
import matplotlib.cm as cm
from src.utils.plotting import make_matching_figure

from src.loftr import LoFTR, default_cfg

def test_image_matching():
    matcher = LoFTR(config=default_cfg)
    matcher.load_state_dict(torch.load("weights/outdoor_ds.ckpt")['state_dict'])
    matcher = matcher.eval()

    # Load example images
    img0_pth = "assets/phototourism_sample_images/london_bridge_78916675_4568141288.jpg"
    img1_pth = "assets/phototourism_sample_images/london_bridge_94185272_3874562886.jpg"
    img0_raw = cv2.imread(img0_pth, cv2.IMREAD_GRAYSCALE)
    img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)
    img0_raw = cv2.resize(img0_raw, (img0_raw.shape[1]//8*8, img0_raw.shape[0]//8*8))  # input size shuold be divisible by 8
    img1_raw = cv2.resize(img1_raw, (img1_raw.shape[1]//8*8, img1_raw.shape[0]//8*8))

    img0 = torch.from_numpy(img0_raw)[None][None] / 255.
    img1 = torch.from_numpy(img1_raw)[None][None] / 255.
    batch = {'image0': img0, 'image1': img1}

    # Inference with LoFTR and get prediction
    with torch.no_grad():
        matcher(batch)
        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()
        mconf = batch['mconf'].cpu().numpy()

    # Draw
    color = cm.jet(mconf)
    text = [
        'LoFTR',
        'Matches: {}'.format(len(mkpts0)),
    ]
    fig = make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, text=text)
    assert len(mkpts0) == len(mkpts1)
    expect0 = np.loadtxt('expect0.txt', dtype=float)
    expect1 = np.loadtxt('expect1.txt', dtype=float)
    for i in range(len(mkpts0)):
        assert tuple(mkpts0[i]) == tuple(expect0[i])
        assert tuple(mkpts1[i]) == tuple(expect1[i])
    print("Success!")
    fig.savefig("demo.png")



test_image_matching()
