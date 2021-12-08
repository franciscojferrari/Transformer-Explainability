import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from numpy import *
import argparse
from PIL import Image
import imageio
import os
import glob

from tqdm import tqdm
from utils.metrices import *

from utils import render


from data.Imagenet import Imagenet_Segmentation
from attribution_generators.ViT_explanation_generator import ExplanationGenerator
from baseline.models.ViT_paper import vit_base_patch16_224 as paper_vit_base_patch16_224
from baseline.models.ViT_ours import vit_base_patch16_224 as our_vit_base_patch16_224

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

import torch.nn.functional as F


def eval_batch(image, labels, evaluator, index, device, lrp, results_dir, experiment_dir, args):
    evaluator.zero_grad()
    # Save input image
    if args.save_img:
        img = image[0].permute(1, 2, 0).data.cpu().numpy()
        img = 255 * (img - img.min()) / (img.max() - img.min())
        img = img.astype('uint8')
        Image.fromarray(
            img, 'RGB').save(
            os.path.join(results_dir, 'input/{}_input.png'.format(index)))
        Image.fromarray(
            (labels.repeat(3, 1, 1).permute(1, 2, 0).data.cpu().numpy() * 255).astype('uint8'),
            'RGB').save(
            os.path.join(results_dir, 'input/{}_mask.png'.format(index)))

    image.requires_grad = True

    image = image.requires_grad_()
    predictions = evaluator(image)

    Res = lrp.generate_LRP(
        image.to(device),
        start_layer=1,
        device=device,
        method="transformer_attribution"
    ).reshape(args.batch_size, 1, 14, 14)

    Res = torch.nn.functional.interpolate(
        Res, scale_factor=16, mode='bilinear', align_corners=False).to(device)

    # threshold between FG and BG is the mean
    Res = (Res - Res.min()) / (Res.max() - Res.min())

    ret = Res.mean()

    Res_1 = Res.gt(ret).type(Res.type())
    Res_0 = Res.le(ret).type(Res.type())

    Res_1_AP = Res
    Res_0_AP = 1-Res

    Res_1[Res_1 != Res_1] = 0
    Res_0[Res_0 != Res_0] = 0
    Res_1_AP[Res_1_AP != Res_1_AP] = 0
    Res_0_AP[Res_0_AP != Res_0_AP] = 0

    # TEST
    pred = Res.clamp(min=args.thr) / Res.max()
    pred = pred.view(-1).data.cpu().numpy()
    target = labels.view(-1).data.cpu().numpy()
    # print("target", target.shape)

    output = torch.cat((Res_0, Res_1), 1)
    output_AP = torch.cat((Res_0_AP, Res_1_AP), 1)

    if args.save_img:
        # Save predicted mask
        mask = F.interpolate(Res_1, [64, 64], mode='bilinear')
        mask = mask[0].squeeze().data.cpu().numpy()
        # mask = Res_1[0].squeeze().data.cpu().numpy()
        mask = 255 * mask
        mask = mask.astype('uint8')
        imageio.imsave(os.path.join(args.exp_img_path, 'mask_' + str(index) + '.jpg'), mask)

        relevance = F.interpolate(Res, [64, 64], mode='bilinear')
        relevance = relevance[0].permute(1, 2, 0).data.cpu().numpy()
        # relevance = Res[0].permute(1, 2, 0).data.cpu().numpy()
        hm = np.sum(relevance, axis=-1)
        maps = (render.hm_to_rgb(hm, scaling=3, sigma=1, cmap='seismic') * 255).astype(np.uint8)
        imageio.imsave(os.path.join(args.exp_img_path, 'heatmap_' + str(index) + '.jpg'), maps)

    # Evaluate Segmentation
    batch_inter, batch_union, batch_correct, batch_label = 0, 0, 0, 0
    batch_ap, batch_f1 = 0, 0

    # Segmentation resutls
    correct, labeled = batch_pix_accuracy(output[0].data.cpu(), labels[0])
    inter, union = batch_intersection_union(output[0].data.cpu(), labels[0], 2)
    batch_correct += correct
    batch_label += labeled
    batch_inter += inter
    batch_union += union
    # print("output", output.shape)
    # print("ap labels", labels.shape)
    # ap = np.nan_to_num(get_ap_scores(output, labels))
    ap = np.nan_to_num(get_ap_scores(output_AP, labels))
    f1 = np.nan_to_num(get_f1_scores(output[0, 1].data.cpu(), labels[0]))
    batch_ap += ap
    batch_f1 += f1

    return batch_correct, batch_label, batch_inter, batch_union, batch_ap, batch_f1, pred, target


def imagenet_seg_dataloader(imagenet_seg_path: str, batch_size: int = 1):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    test_img_trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    test_lbl_trans = transforms.Compose([
        transforms.Resize((224, 224), Image.NEAREST),
    ])

    ds = Imagenet_Segmentation(imagenet_seg_path,
                               transform=test_img_trans, target_transform=test_lbl_trans)

    sample_loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=50,
        pin_memory=True,
        drop_last=False
    )
    return sample_loader


def run_seg_eval(args):

    directory = os.path.join(args.work_path, 'run')
    runs = sorted(glob.glob(os.path.join(directory, 'experiment_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

    experiment_dir = os.path.join(directory, 'experiment_{}'.format(str(run_id)))

    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    results_dir = os.path.join(experiment_dir, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(os.path.join(results_dir, 'input')):
        os.makedirs(os.path.join(results_dir, 'input'))
    if not os.path.exists(os.path.join(results_dir, 'explain')):
        os.makedirs(os.path.join(results_dir, 'explain'))

    args.exp_img_path = os.path.join(results_dir, 'explain/img')
    if not os.path.exists(args.exp_img_path):
        os.makedirs(args.exp_img_path)
    args.exp_np_path = os.path.join(results_dir, 'explain/np')
    if not os.path.exists(args.exp_np_path):
        os.makedirs(args.exp_np_path)

    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")

    if args.vit_model == "ours":
        model = our_vit_base_patch16_224().to(device)
    else:
        model = paper_vit_base_patch16_224().to(device)

    attribution_generator = ExplanationGenerator(model)
    model.eval()

    total_inter, total_union, total_correct, total_label = np.int64(
        0), np.int64(0), np.int64(0), np.int64(0)
    total_ap, total_f1 = [], []

    predictions, targets = [], []

    imagenet_seg = imagenet_seg_dataloader(args.imagenet_seg_path, args.batch_size)

    iterator = tqdm(imagenet_seg)

    for batch_idx, (image, labels) in enumerate(iterator):

        images = image.to(device)
        labels = labels.to(device)
        # print("image", image.shape)
        # print("lables", labels.shape)

        correct, labeled, inter, union, ap, f1, pred, target = eval_batch(
            images, labels, model, batch_idx, device, attribution_generator, results_dir, experiment_dir, args)

        predictions.append(pred)
        targets.append(target)

        total_correct += correct.astype('int64')
        total_label += labeled.astype('int64')
        total_inter += inter.astype('int64')
        total_union += union.astype('int64')
        total_ap += [ap]
        total_f1 += [f1]
        pixAcc = np.float64(1.0) * total_correct / (np.spacing(1, dtype=np.float64) + total_label)
        IoU = np.float64(1.0) * total_inter / (np.spacing(1, dtype=np.float64) + total_union)
        mIoU = IoU.mean()
        mAp = np.mean(total_ap)
        mF1 = np.mean(total_f1)
        iterator.set_description(
            'pixAcc: %.4f, mIoU: %.4f, mAP: %.4f, mF1: %.4f' % (pixAcc, mIoU, mAp, mF1))

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    pr, rc, thr = precision_recall_curve(targets, predictions)
    np.save(os.path.join(experiment_dir, 'precision.npy'), pr)
    np.save(os.path.join(experiment_dir, 'recall.npy'), rc)

    plt.figure()
    plt.plot(rc, pr)
    plt.savefig(os.path.join(experiment_dir, 'PR_curve_{}.png'.format(args.method)))

    txtfile = os.path.join(experiment_dir, 'result_mIoU_%.4f.txt' % mIoU)
    # txtfile = 'result_mIoU_%.4f.txt' % mIoU
    fh = open(txtfile, 'w')
    print("Mean IoU over %d classes: %.4f\n" % (2, mIoU))
    print("Pixel-wise Accuracy: %2.2f%%\n" % (pixAcc * 100))
    print("Mean AP over %d classes: %.4f\n" % (2, mAp))
    print("Mean F1 over %d classes: %.4f\n" % (2, mF1))

    fh.write("Mean IoU over %d classes: %.4f\n" % (2, mIoU))
    fh.write("Pixel-wise Accuracy: %2.2f%%\n" % (pixAcc * 100))
    fh.write("Mean AP over %d classes: %.4f\n" % (2, mAp))
    fh.write("Mean F1 over %d classes: %.4f\n" % (2, mF1))
    fh.close()


if __name__ == "__main__":
    # hyperparameters
    num_workers = 0

    # Args
    parser = argparse.ArgumentParser(description='Training multi-class classifier')

    parser.add_argument('--thr', type=float, default=0.,
                        help='threshold')

    parser.add_argument('--save-img', action='store_true',
                        default=False,
                        help='')

    parser.add_argument('--vit-model', type=str,
                        # required=True,
                        default="ours",
                        help='ours or paper')

    parser.add_argument('--imagenet-seg-path', type=str,
                        # required=True,
                        default="/home/tf-exp-o-data/gtsegs_ijcv.mat",
                        )
    parser.add_argument('--work-path', type=str,
                        # required=True,
                        default="/home/tf-exp-o-data/",
                        help='')
    _args = parser.parse_args()
    _args.batch_size = 1

    run_seg_eval(_args)
