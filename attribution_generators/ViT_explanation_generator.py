import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.transform import resize
from tqdm import tqdm

class ExplanationGenerator:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_LRP(
            self, input, index=None, method="transformer_attribution", is_ablation=False,
            start_layer=0, device="cuda"):
        output = self.model(input)
        kwargs = {"alpha": 1}
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(device) * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        R = self.model.relprop(torch.tensor(one_hot_vector).to(device), method=method,
                               is_ablation=is_ablation, start_layer=start_layer,
                               device=device, **kwargs)

        if method == 'lrp':
            R = R.reshape(input.shape[0], 1, 224, 224)
        else:
            R = R.reshape(input.shape[0], 1, 14, 14)
            R = torch.nn.functional.interpolate(R, scale_factor=16, mode='bilinear').to(device)

        return R

    def generate_cam_attn(self, input, index=None, device="cuda"):
        output = self.model(input)
        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(device) * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        # attn
        grad = self.model.blocks[-1].attn.get_attn_gradients()
        cam = self.model.blocks[-1].attn.get_attn()
        cam = cam[0, :, 0, 1:].reshape(-1, 14, 14)
        grad = grad[0, :, 0, 1:].reshape(-1, 14, 14)
        grad = grad.mean(dim=[1, 2], keepdim=True)
        cam = (cam * grad).mean(0).clamp(min=0)
        cam = (cam - cam.min()) / (cam.max() - cam.min())

        cam = cam.reshape(input.shape[0], 1, 14, 14)
        cam = torch.nn.functional.interpolate(cam, scale_factor=16, mode='bilinear').to(device)
        return cam
      
      
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

        with torch.no_grad():
            if class_idx is None:
                pred = self.model(x.data)
                class_idx = torch.argmax(pred).item()

        # DM: for each mask, do inference and then update with the class probability
        sal = torch.zeros((H, W)).to(self.device)
        with torch.no_grad():
            for n in range(0, N, self.gpu_batch):
                masks_ = self.masks[n:min(n + self.gpu_batch, N)]               # (B, 1, H, W)
                masked_img = torch.mul(masks_, x.data)                          # (B, C, H, W)
                logits = self.model(masked_img)                                 # (B, 1000)
                prob = F.softmax(logits, dim=1)[:, class_idx]                   # (B)
                weighted_masks = prob.reshape(-1,1,1) * masks_.squeeze()        # (B, H, W)
                sal += weighted_masks.sum(axis=0)                               # (H, W)
            
        sal = sal / N / self.p1
        return sal.squeeze()



