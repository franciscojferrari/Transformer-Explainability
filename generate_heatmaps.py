from baseline.models.ViT_paper import vit_base_patch16_224 as paper_vit_base_patch16_224
from baseline.models.ViT_ours import vit_base_patch16_224 as our_vit_base_patch16_224
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
from tqdm import tqdm
from attribution_generators.ViT_explanation_generator import ExplanationGenerator
import h5py
import os
from torchvision.datasets import ImageNet
from argparse import ArgumentParser
import logger


def compute_saliency_and_save(images, path, expl_gen, device):
    first = True

    try:
        os.remove(os.path.join(path, 'results_new.hdf5'))
    except OSError:
        pass

    with h5py.File(os.path.join(path, 'results_new.hdf5'), 'a') as f:
        data_cam = f.create_dataset('vis',
                                    (1, 1, 224, 224),
                                    maxshape=(None, 1, 224, 224),
                                    dtype=np.float32,
                                    compression="gzip")
        data_image = f.create_dataset('image',
                                      (1, 3, 224, 224),
                                      maxshape=(None, 3, 224, 224),
                                      dtype=np.float32,
                                      compression="gzip")
        data_target = f.create_dataset('target',
                                       (1,),
                                       maxshape=(None,),
                                       dtype=np.int32,
                                       compression="gzip")
        for batch_idx, (data, target) in enumerate(tqdm(images)):
            if first:
                first = False
                data_cam.resize(data_cam.shape[0] + data.shape[0] - 1, axis=0)
                data_image.resize(data_image.shape[0] + data.shape[0] - 1, axis=0)
                data_target.resize(data_target.shape[0] + data.shape[0] - 1, axis=0)
            else:
                data_cam.resize(data_cam.shape[0] + data.shape[0], axis=0)
                data_image.resize(data_image.shape[0] + data.shape[0], axis=0)
                data_target.resize(data_target.shape[0] + data.shape[0], axis=0)

            # Add data
            data_image[-data.shape[0]:] = data.data.cpu().numpy()
            data_target[-data.shape[0]:] = target.data.cpu().numpy()

            target = target.to(device)
            data = data.to(device)
            data.requires_grad_()

            index = None
            if args.method == 'attn_gradcam':
                Res = expl_gen.generate_cam_attn(data,  index=index, device=device)
            else:
                Res = expl_gen.generate_LRP(
                    data, start_layer=1, method=args.method, index=index, device=device,
                    use_1_3=args.use_1_3, use_eps_rule=args.use_eps)

            Res = (Res - Res.min()) / (Res.max() - Res.min())

            data_cam[-data.shape[0]:] = Res.data.cpu().numpy()


def imagenet_dataloader(imagenet_validation_path: str, batch_size: int = 1, NCC=False):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    if NCC:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    imagenet_ds = ImageNet(imagenet_validation_path, split='val', transform=transform)
    sample_loader = torch.utils.data.DataLoader(
        imagenet_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    return sample_loader


@logger.timed
def generate_heatmaps(args):
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")

    if args.vit_model == "ours":
        logger.logger.debug("Using OUR ViT")
        model = our_vit_base_patch16_224().to(device)
    else:
        logger.logger.debug("Using PAPER ViT")
        model = paper_vit_base_patch16_224().to(device)

    model.eval()
    attribution_generator = ExplanationGenerator(model)
    imagenet = imagenet_dataloader(args.imagenet_validation_path, args.batch_size, args.NCC)

    compute_saliency_and_save(imagenet, args.save_path, attribution_generator, device)


if __name__ == "__main__":
    parser = ArgumentParser(description='Train a segmentation')
    parser.add_argument('--batch-size', type=int,
                        default=1,
                        help='')
    parser.add_argument('--work-path', type=str,
                        # required=True,
                        default="/home/tf-exp-o-data/",
                        help='')
    parser.add_argument('--vit-model', type=str,
                        # required=True,
                        default="ours",
                        help='ours or paper')
    parser.add_argument('--method', type=str,
                        # required=True,
                        default="transformer_attribution",
                        help='')
    parser.add_argument('--use-1-3',
                        # required=True,
                        default=False,
                        action='store_true',
                        help='')
    parser.add_argument('--NCC',
                        # required=True,
                        default=False,
                        action='store_true',
                        help='')

    parser.add_argument('--use-eps',
                        # required=True,
                        default=True,
                        action='store_true',
                        help='')

    args = parser.parse_args()
    args.imagenet_validation_path = os.path.join(args.work_path, "imgnet_val")

    mthd = args.method + "_13" if args.use_1_3 else args.method
    mthd = args.method + "_NCC" if args.NCC else args.method
    mthd = args.method + "_use_eps" if args.use_eps else args.method

    args.save_path = os.path.join(args.work_path, "results", args.vit_model, mthd)
    os.makedirs(args.save_path, exist_ok=True)

    assert args.vit_model == "ours" or args.vit_model == "paper", "please select ours or paper"
    generate_heatmaps(args)

# python generate_heatmaps.py --imagenet-validation-path /home/tf-exp-o-data/imgnet_val/ --save-path /home/tf-exp-o-data/results/ --vit-model ours
