import numpy as np
import torch


class LRP:
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

        R = self.model.relprop(torch.tensor(one_hot_vector).to(device),method=method, 
            is_ablation=is_ablation, start_layer=start_layer,
            device=device, **kwargs)

        if method == 'partial_lrp':
            R = R.reshape(input.shape[0], 1, 224, 224)
        else:
            R = R.reshape(input.shape[0], 1, 14, 14)
            R = torch.nn.functional.interpolate(R, scale_factor=16, mode='bilinear').to(device)

        return R
