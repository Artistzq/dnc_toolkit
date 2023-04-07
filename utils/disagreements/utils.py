import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import cv2
import math

def find_agreements(model1, model2, number, test_loader):
    """返回number张在test_loader中，model1和model2分类一致的图片。

    Args:
        model1 (torch.nn.Module): _description_
        model2 (torch.nn.Module): _description_
        number (int): _description_
        test_loader (torch.utils.data.DataLoader): _description_

    Returns:
        Tuple(Tensor, Tensor): _description_
    """
    same_indices = []
    same_images = []
    same_labels = []
    for i, (images, label) in enumerate(test_loader):
        images = images.to("cuda")
        label  = label.to("cuda")
        pred_m = torch.argmax(model1(images), dim=-1);
        pred_t = torch.argmax(model2(images), dim=-1);
        # print(pred_m- pred_t)
        indices = torch.where(pred_m == pred_t)[0]
        # indices += i * 128
        same_indices.append(indices)
        same_images.append(images[indices, :, :, :])
        same_labels.append(label[indices])
        # break
    same_images = torch.cat(same_images)
    same_labels = torch.cat(same_labels)
    return same_images[: number], same_labels[: number]


def generate_disagreements(image_loader, attack):
    """返回imageloader中的图片经过攻击后的图片，攻击效果为两个模型分类结果不同

    Args:
        image_loader (_type_): _description_
        attack (_type_): _description_

    Returns:
        _type_: _description_
    """
    adv_images = []
    raw_labels = []
    for images, labels in image_loader:
        adv_images.append(attack(images, labels))
        raw_labels.append(labels)
    adv_images = torch.cat(adv_images)
    raw_labels = torch.cat(raw_labels)
    return adv_images, raw_labels


def tensor_to_loader(images, labels, batch_size=100):
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size, shuffle=False)


def disagreement_rate(model1, model2, loader):
    dif = 0
    total = 0
    for X, y in loader:
        X = X.to("cuda")
        pred_m = torch.argmax(model1(X), dim=-1)
        pred_t = torch.argmax(model2(X), dim=-1)
        dif += torch.count_nonzero(pred_m - pred_t)
        total += pred_m.shape[0]
    return dif.item() * 100 / total
