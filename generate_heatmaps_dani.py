from baseline.models.ViT_paper import vit_base_patch16_224 as paper_vit_base_patch16_224
from baseline.models.ViT_ours import vit_base_patch16_224 as our_vit_base_patch16_224
from baseline.models.ViT_original import vit_base_patch16_224 as original_base_model
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
from tqdm import tqdm
from attribution_generators.ViT_explanation_generator import ExplanationGenerator, RISE
import h5py
import os
from torchvision.datasets import ImageNet
from argparse import ArgumentParser
#import logger


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam


def generate_visualization(original_image, attribution_generator, class_index=None, device="cuda"):
    transformer_attribution = attribution_generator.generate_LRP(
        original_image.unsqueeze(0).to(device),
        method="transformer_attribution", index=class_index,
        device=device
    ).detach()

    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    transformer_attribution = torch.nn.functional.interpolate(
        transformer_attribution, scale_factor=16, mode='bilinear', align_corners=False)
    transformer_attribution = transformer_attribution.reshape(
        224, 224).to(device).data.cpu().numpy()

    transformer_attribution = (transformer_attribution - transformer_attribution.min()
                               ) / (transformer_attribution.max() - transformer_attribution.min())
    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (
        image_transformer_attribution - image_transformer_attribution.min()) / (
        image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


def compute_saliency_and_save(images, path, lrp, rise, device):
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

            if args.method == 'rise':
                Res = rise(data)
            else:
                Res = lrp.generate_LRP(data, start_layer=1, method=args.method,
                                       index=index, device=device)

            Res = (Res - Res.min()) / (Res.max() - Res.min())

            data_cam[-data.shape[0]:] = Res.data.cpu().numpy()


def imagenet_dataloader(imagenet_validation_path: str, batch_size: int = 1):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    imagenet_ds = ImageNet(imagenet_validation_path, split='val',
                           download=False, transform=transform)
    sample_loader = torch.utils.data.DataLoader(
        imagenet_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    return sample_loader


# @logger.timed
def generate_heatmaps(args):
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    if args.vit_model == "ours":
        #logger.logger.debug("Using OUR ViT")
        model = our_vit_base_patch16_224().to(device)
    elif args.vit_model == "original":
        model = original_base_model().to(device)
    else:
        #logger.logger.debug("Using PAPER ViT")
        model = paper_vit_base_patch16_224().to(device)

    model.eval()

    attribution_generator = ExplanationGenerator(model)

    rise = None
    if args.method == 'rise':
        rise = RISE(model, (224, 224), device)
        rise.generate_masks(N=2000, s=8, p1=0.5)

    imagenet = imagenet_dataloader(args.imagenet_validation_path, args.batch_size)

    #results_path = os.path.join(args.method_dir, args.vit_model)

    compute_saliency_and_save(imagenet, args.method_dir, attribution_generator, rise, device)


def main_test():
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    # initialize ViT pretrained

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    # model = vit_base_patch16_224().to(device)
    model = our_vit_base_patch16_224().to(device)
    model.eval()
    attribution_generator = ExplanationGenerator(model)
    image = Image.open('samples/catdog.png')
    dog_cat_image = transform(image)

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(image)
    axs[0].axis('off')

    output = model(dog_cat_image.unsqueeze(0).to(device))

    visualize = False

    if visualize:
        # cat - the predicted class
        cat = generate_visualization(
            dog_cat_image, attribution_generator=attribution_generator, device=device)

        # dog
        # generate visualization for class 243: 'bull mastiff'
        dog = generate_visualization(
            dog_cat_image, attribution_generator=attribution_generator, class_index=243,
            device=device)

        axs[1].imshow(cat)
        axs[1].axis('off')
        axs[2].imshow(dog)
        axs[2].axis('off')
        plt.show()
    else:
        images = []
        image = Image.open('samples/dogbird.png')
        dog_bird_image = transform(image)

        images.append((dog_cat_image.unsqueeze(0), torch.tensor(282)))
        # images.append((dog_cat_image, 243))
        images.append((dog_bird_image.unsqueeze(0), torch.tensor(161)))
        # images.append((dog_bird_image, 87))
        compute_saliency_and_save(images, "results", attribution_generator, device)


if __name__ == "__main__":
    parser = ArgumentParser(description='Train a segmentation')
    parser.add_argument('--batch-size', type=int,
                        default=1,
                        help='')
    parser.add_argument('--work-path', type=str,
                        # required=True,
                        default="/home/tf-exp-o-data/",
                        help='')
    parser.add_argument('--imagenet-validation-path', type=str,
                        required=True,
                        # default="/home/tf-exp-o-data/",
                        help='')
    parser.add_argument('--vit-model', type=str,
                        # required=True,
                        default="paper",
                        help='ours or paper')
    parser.add_argument('--method', type=str,
                        # required=True,
                        default="transformer_attribution",
                        help='')

    args = parser.parse_args()
    #args.imagenet_validation_path = os.path.join(args.work_path, "imgnet_val")
    #args.save_path = os.path.join(args.work_path, "results")

    # PATH variables
    PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
    os.makedirs(os.path.join(PATH, 'visualizations'), exist_ok=True)

    try:
        os.remove(os.path.join(PATH, 'visualizations/my_results/results.hdf5'))
    except OSError:
        pass

    ablation_fold = 'not_ablation'
    os.makedirs(os.path.join(PATH, 'visualizations/my_results'), exist_ok=True)
    args.method_dir = os.path.join(PATH, 'visualizations/my_results')

    assert args.vit_model == "ours" or args.vit_model == "paper" or args.vit_model == "original", "please select ours or paper"

    generate_heatmaps(args)

# python generate_heatmaps.py --imagenet-validation-path /home/tf-exp-o-data/imgnet_val/ --save-path /home/tf-exp-o-data/results/ --vit-model ours
