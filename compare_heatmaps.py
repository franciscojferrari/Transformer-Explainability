from baseline.models.ViT_paper import vit_base_patch16_224 as paper_base_model
from baseline.models.ViT_ours import vit_base_patch16_224 as our_base_model
from baseline.models.ViT_original import vit_base_patch16_224 as original_base_model
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
import pdb
from attribution_generators.ViT_explanation_generator import ExplanationGenerator
import h5py
import os
import time
from attribution_generators.rise import RISE
from cls2idx import CLS2IDX


def print_top_classes(predictions, **kwargs):
    # Print Top-5 predictions
    prob = torch.softmax(predictions, dim=1)
    class_indices = predictions.data.topk(5, dim=1)[1][0].tolist()
    max_str_len = 0
    class_names = []
    for cls_idx in class_indices:
        class_names.append(CLS2IDX[cls_idx])
        if len(CLS2IDX[cls_idx]) > max_str_len:
            max_str_len = len(CLS2IDX[cls_idx])

    print('Top 5 classes:')
    for cls_idx in class_indices:
        output_string = '\t{} : {}'.format(cls_idx, CLS2IDX[cls_idx])
        output_string += ' ' * (max_str_len - len(CLS2IDX[cls_idx])) + '\t\t'
        output_string += 'value = {:.3f}\t prob = {:.1f}%'.format(
            predictions[0, cls_idx], 100 * prob[0, cls_idx])
        print(output_string)


def normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    return tensor


def gen_raw_attr(base_model, image, device, model_name=None):
    model = base_model().to(device)
    model.eval()
    attr_gen = ExplanationGenerator(model)

    output = model(image.unsqueeze(0).to(device))
    print_top_classes(output)

    start_time = time.time()
    raw_attr = attr_gen.generate_LRP(
        image.unsqueeze(0).to(device), method="transformer_attribution", device=device
    ).detach()
    print(model_name)
    print('Attention computation exec time:\t%s sec' % (time.time() - start_time))

    return raw_attr


def get_heatmap(raw_attr, max_=None):
    raw_attr = raw_attr.reshape(1, 1, 14, 14)
    attr = torch.nn.functional.interpolate(
        raw_attr, scale_factor=16, mode='bilinear', align_corners=False)
    attr = attr.reshape(224, 224).data.cpu().numpy()

    if max_ is None:
        max_ = attr.max()

    attr = (attr - attr.min()) / (max_ - attr.min())
    heatmap = cv2.applyColorMap(np.uint8(255 * attr), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.cvtColor(np.array(heatmap), cv2.COLOR_RGB2BGR)
    return heatmap


def main():
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    image = Image.open('samples/catdog.png')
    dog_cat_image = transform(image)

    '''
    # this is to generate a heatmap with RISE
    model = original_base_model().to(device)
    model.eval()    
    rise = RISE(model, input_size=(224, 224), device=device, gpu_batch=100)
    rise.generate_masks(N=2000, s=8, p1=0.5)
    #heatmap_rise = rise(dog_cat_image.unsqueeze(0).to(device), class_idx=243)
    heatmap_rise = rise(dog_cat_image.unsqueeze(0).to(device))

    '''
    raw_attr_paper = gen_raw_attr(paper_base_model, dog_cat_image,
                                  device, model_name='Original paper implementation')
    raw_attr_ours = gen_raw_attr(our_base_model, dog_cat_image, device,
                                 model_name='Our implementation')

    if torch.equal(raw_attr_paper, raw_attr_ours):
        print('\n*** The two attribution maps ARE the same ***\n')
    else:
        print('\n*** The two attribution maps ARE NOT the same ***\n')

    heatmap_paper = get_heatmap(raw_attr_paper)
    heatmap_ours = get_heatmap(raw_attr_ours)
    #max_attr = np.maximum(raw_attr_paper.max(), raw_attr_ours.max())
    #heatmap_diff = get_heatmap(np.abs(raw_attr_paper - raw_attr_ours), max_=max_attr)
    #heatmap_diff_augmented = get_heatmap(np.abs(raw_attr_paper - raw_attr_ours))

    fig, axs = plt.subplots(1, 5)
    axs[0].imshow(image)
    axs[0].axis('off')
    axs[0].title.set_text('Original image')
    # axs[1].imshow(heatmap_rise)
    # axs[1].axis('off')

    axs[1].imshow(heatmap_paper)
    axs[1].axis('off')
    axs[1].title.set_text('Paper\'s heatmap')
    axs[2].imshow(heatmap_ours)
    axs[2].axis('off')
    axs[2].title.set_text('Our heatmap')
    '''
    axs[3].imshow(heatmap_diff)
    axs[3].axis('off')
    axs[3].title.set_text('Heatmap diff')
    axs[4].imshow(heatmap_diff_augmented)
    axs[4].axis('off')
    axs[4].title.set_text('Heatmap diff augmented %.6f' % max_attr)
    '''
    plt.show()


if __name__ == "__main__":
    main()
