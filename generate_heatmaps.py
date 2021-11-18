from baseline.models.ViT_base import vit_base_patch16_224
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2


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


def main():
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

    model = vit_base_patch16_224().to_device(device)
    model.eval()

    image = Image.open(input("Insert Path: "))
    dog_cat_image = transform(image)

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(image)
    axs[0].axis('off')

    output = model(dog_cat_image.unsqueeze(0).to_device(device))
    print_top_classes(output)


if __name__ == "__main__":
    main()
