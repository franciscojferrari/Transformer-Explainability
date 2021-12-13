import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.transform import resize
from tqdm import tqdm
import pdb

class RISE(nn.Module):
    def __init__(self, model, input_size, device, gpu_batch=100):
        super(RISE, self).__init__()
        self.model = model
        self.input_size = input_size
        self.gpu_batch = gpu_batch
        self.device = device

    def generate_masks(self, N, s, p1):
        cell_size = np.ceil(np.array(self.input_size) / s)
        up_size = (s + 1) * cell_size

        grid = np.random.rand(N, s, s) < p1
        grid = grid.astype('float32')

        self.masks = np.empty((N, *self.input_size))

        for i in tqdm(range(N), desc='Generating filters'):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            self.masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                         anti_aliasing=False)[x:x + self.input_size[0], y:y + self.input_size[1]]
        self.masks = self.masks.reshape(-1, 1, *self.input_size)
        self.masks = torch.from_numpy(self.masks).float()
        self.masks = self.masks.to(self.device)
        self.N = N
        self.p1 = p1

    def forward(self, x, class_idx=None):
        N = self.N
        _, _, H, W = x.size()

        if class_idx is None:
            pred = self.model(x.data)
            class_idx = torch.argmax(pred).item()

        # DM: for each mask, do inference and then update with the class probability
        sal = torch.zeros((H, W)).to(self.device)
        with torch.no_grad():
            for n in tqdm(range(0, N, self.gpu_batch)):
                masks_ = self.masks[n:min(n + self.gpu_batch, N)]               # (B, 1, H, W)
                masked_img = torch.mul(masks_, x.data)                          # (B, C, H, W)
                logits = self.model(masked_img)                                 # (B, 1000)
                prob = F.softmax(logits, dim=1)[:, class_idx]                   # (B)
                weighted_masks = prob.reshape(-1,1,1) * masks_.squeeze()        # (B, H, W)
                sal += weighted_masks.sum(axis=0)                               # (H, W)
            
        sal = sal / N / self.p1
        return sal.squeeze().cpu().detach().numpy()