import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data


''' FROM TOFLOW GITHUB'''
class MemoryFriendlyLoader(torch.utils.data.Dataset):
    def __init__(self, origin_img_dir, pathlistfile, edited_img_dir='', task=''):
        self.origin_img_dir = origin_img_dir
        self.edited_img_dir = edited_img_dir
        self.task = task
        self.pathlist = self.loadpath(pathlistfile)
        self.count = len(self.pathlist)

    def loadpath(self, pathlistfile):
        fp = open(pathlistfile)
        pathlist = fp.read().splitlines()
        fp.close()
        return pathlist

    def __getitem__(self, index):
        frames = []
        path_code = self.pathlist[index]
        if self.task == 'interp':
            N = 2   # 这里的N仅仅是为了下面取framex方便, 并非是论文里的N
            for i in [1, 3]:
                frames.append(plt.imread(os.path.join(self.origin_img_dir, path_code, 'im%d.png' % i)))                  # load the first and third images
            frames.append(plt.imread(os.path.join(self.origin_img_dir, path_code, 'im2.png')))                           # load ground truth (the second one)
        else:
            N = 7
            for i in range(7):
                frames.append(plt.imread(os.path.join(self.edited_img_dir, path_code, 'im%04d.png' % (i + 1))))          # load images with noise.
            frames.append(plt.imread(os.path.join(self.origin_img_dir, path_code, 'im4.png')))                           # load ground truth

        frames = np.array(frames)
        framex = np.transpose(frames[0:N, :, :, :], (0, 3, 1, 2))
        framey = np.transpose(frames[-1, :, :, :], (2, 0, 1))

        return torch.from_numpy(framex), torch.from_numpy(framey), path_code

    def __len__(self):
        return self.count
''' END '''

''' From SOF-VSR Paper '''
class SOFVSR(nn.Module):
    def __init__(self, cfg, n_frames=3, is_training=True):
        super(SOFVSR, self).__init__()
        self.scale = cfg.scale
        self.is_training = is_training
        self.OFR = OFRnet(scale=cfg.scale, channels=320)
        self.SR = SRnet(scale=cfg.scale, channels=320, n_frames=n_frames)

    def forward(self, x):
        b, n_frames, c, h, w = x.size()     # x: b*n*c*h*w
        idx_center = (n_frames - 1) // 2

        # motion estimation
        flow_L1 = []
        flow_L2 = []
        flow_L3 = []
        input = []

        for idx_frame in range(n_frames):
            if idx_frame != idx_center:
                input.append(torch.cat((x[:,idx_frame,:,:,:], x[:,idx_center,:,:,:]), 1))
        optical_flow_L1, optical_flow_L2, optical_flow_L3 = self.OFR(torch.cat(input, 0))

        optical_flow_L1 = optical_flow_L1.view(-1, b, 2, h//2, w//2)
        optical_flow_L2 = optical_flow_L2.view(-1, b, 2, h, w)
        optical_flow_L3 = optical_flow_L3.view(-1, b, 2, h*self.scale, w*self.scale)

        # motion compensation
        draft_cube = []
        draft_cube.append(x[:, idx_center, :, :, :])

        for idx_frame in range(n_frames):
            if idx_frame == idx_center:
                flow_L1.append([])
                flow_L2.append([])
                flow_L3.append([])
            if idx_frame != idx_center:
                if idx_frame < idx_center:
                    idx = idx_frame
                if idx_frame > idx_center:
                    idx = idx_frame - 1

                flow_L1.append(optical_flow_L1[idx, :, :, :, :])
                flow_L2.append(optical_flow_L2[idx, :, :, :, :])
                flow_L3.append(optical_flow_L3[idx, :, :, :, :])

                for i in range(self.scale):
                    for j in range(self.scale):
                        draft = optical_flow_warp(x[:, idx_frame, :, :, :],
                                                  optical_flow_L3[idx, :, :, i::self.scale, j::self.scale] / self.scale)
                        draft_cube.append(draft)
        draft_cube = torch.cat(draft_cube, 1)

        # super-resolution
        SR = self.SR(draft_cube)

        if self.is_training:
            return flow_L1, flow_L2, flow_L3, SR
        if not self.is_training:
            return SR



''' END '''