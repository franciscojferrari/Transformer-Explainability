import torch
import os
from tqdm import tqdm
import numpy as np
import glob
from baseline.models.ViT_ours import vit_base_patch16_224 as our_vit_base_patch16_224
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
from tqdm import tqdm
from attribution_generators.ViT_explanation_generator import ExplanationGenerator
import os
from torchvision.datasets import ImageNet
import re
from cls2idx import CLS2IDX


def imagenet_dataloader(imagenet_validation_path: str, batch_size: int = 1):\

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        #         normalize,
    ])

    imagenet_ds = ImageNet(imagenet_validation_path, split='val',
                           download=False, transform=transform)
    sample_loader = torch.utils.data.DataLoader(
        imagenet_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    return sample_loader


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam


def generate_visualization(
        original_image, attribution_generator, class_index=None, device="cuda",
        method="transformer_attribution"):

    if method == 'attn_gradcam':
        transformer_attribution = attribution_generator.generate_cam_attn(
            original_image.unsqueeze(0).to(device),
            index=class_index,
            device=device
        ).detach()
    else:
        transformer_attribution = attribution_generator.generate_LRP(
            original_image.unsqueeze(0).to(device),
            method=method, index=class_index,
            device=device
        ).detach()

    transformer_attribution = (transformer_attribution - transformer_attribution.min()
                               ) / (transformer_attribution.max() - transformer_attribution.min())
    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (
        image_transformer_attribution - image_transformer_attribution.min()) / (
        image_transformer_attribution.max() - image_transformer_attribution.min())

    transformer_attribution = transformer_attribution[0].permute(1, 2, 0)
    transformer_attribution = transformer_attribution.data.cpu().numpy()

    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


if __name__ == "__main__":
    method_columns = {'Input': 'Input', 'attn_gradcam': "GradCAM", 'lrp': "LRP",
                      'partial_lrp': "partial LPR", 'rollout': "Rollout",
                      'transformer_attribution': "Ours"}

    fig, axs = plt.subplots(4, len(method_columns), figsize=(15, 10))
    dataloader = imagenet_dataloader("/home/tf-exp-o-data/imgnet_val/", 10)

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    model = our_vit_base_patch16_224().to(device)
    model.eval()
    attribution_generator = ExplanationGenerator(model)

    # imgs, targets = next(iter(dataloader))
    counter = 20
    for imgs, targets in tqdm(dataloader):
        idx = 0
        for _, (img, y) in enumerate(zip(imgs, targets)):

            norm_img = normalize(img)
            output = model(norm_img.unsqueeze(0).to(device))
            index = np.argmax(output.cpu().data.numpy(), axis=-1)[0].astype(int)
            y = y.cpu().item()

            if y != index:
                print(CLS2IDX[y], CLS2IDX[index])
                continue
            if idx == len(axs):
                break

            for idc, (method, column) in enumerate(method_columns.items()):
                if idx == 0:
                    axs[idx][idc].set_title(column)
                if idc == 0:
                    axs[idx][idc].set_ylabel(CLS2IDX[y].split(",")[0])
                if column == "Input":
                    axs[idx][idc].imshow(img.permute(1, 2, 0).data.cpu().numpy())
                else:
                    vis = generate_visualization(
                        img, attribution_generator=attribution_generator, device=device,
                        method=method)
                    axs[idx][idc].imshow(vis)

                # Turn off tick labels
                axs[idx][idc].set_yticklabels([])
                axs[idx][idc].set_xticklabels([])
                axs[idx][idc].set_xticks([])
                axs[idx][idc].set_yticks([])
            idx += 1

        images = sorted(glob.glob(os.path.join("images", '*.png')),
                        key=lambda f: int(re.sub('\D', '', f)))
        images_id = int(os.path.splitext(os.path.basename(images[-1]))[0]) + 1 if images else 0
        fig.savefig(
            os.path.join("images", '{}.png'.format(str(images_id))),
            bbox_inches='tight', pad_inches=0)

        counter -= 1
        if counter == 0:
            break
