import numpy as np
import os
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Normalize, ToTensor
import torch


full_categories = [
    'jpeg_compression',
    'shot_noise',
    'elastic_transform',
    'glass_blur',
    'zoom_blur',
    'impulse_noise',
    'speckle_noise',
    'pixelate',
    'motion_blur',
    'gaussian_blur',
    'frost',
    'defocus_blur',
    'fog',
    'snow',
    'brightness',
    'saturate',
    'gaussian_noise',
    'contrast',
    'spatter'
]


def get_corrupt_dataset(root, dataset_name, serverity, as_loader=True, ood_categories=None, batch_size=100, normalization=None):
    """
    1. 获取全部数据集，以loader形式
    >>> test_loaders = get_corrupt_dataset(root="./data", dataset_name="CIFAR10", serverity=1, batch_size=128)
    >>> test_loaders.keys()
    >>> dict_keys(['jpeg_compression', 'shot_noise', 'elastic_transform', 'glass_blur', 'zoom_blur', 
            'impulse_noise', 'speckle_noise', 'pixelate', 'motion_blur', 'gaussian_blur', 'frost', 
            'defocus_blur', 'fog', 'snow', 'brightness', 'saturate', 'gaussian_noise', 'contrast', 'spatter'])
            
    >>> images, labels = test_loaders["speckle_noise"].__iter__().next()
    >>> images.shape, labels.shape
    >>> (torch.Size([128, 3, 32, 32]), torch.Size([128]))
    
    2. 获取指定污染类型的数据集，以Torch Dataset形式
    >>> test_sets = get_corrupt_dataset(root="./data", dataset_name="TinyImageNet", serverity=2, as_loader=False,
                    ood_categories=["jpeg_compression","shot_noise"], normalization=self_normalization)
    >>> single_image, single_label = test_sets["jpeg_compression"][0], test_sets["jpeg_compression"][1] 
    >>> single_image.shape, single_label.shape
    >>> (torch.Size([3, 32, 32]), torch.Size([]))
    
    Args:
        root (str): 数据集目录
        dataset_name (str): 数据集名称，CIFAR10 / CIFAR100 / TinyImageNet
        serverity (int): 损坏程度
        as_loader (bool, optional): 是否返回loader. Defaults to True.
        ood_categories (list or tuple, optional): 哪些变种. Defaults to 全部.
        batch_size (int, optional): Defaults to 100.
        normalization (list or typle, optional): Defaults to 默认标准化.

    Returns:
        _type_: _description_
    """
    assert dataset_name in ["CIFAR10", "CIFAR100", "TinyImageNet"], "Dataset Name Wrong."
    
    # 种类
    if not ood_categories:
        ood_categories = full_categories

    # 标准化参数
    if normalization is None:
        if dataset_name == 'CIFAR100':
            normalization = ([0.507, 0.487, 0.441], [0.267, 0.256, 0.276])
        elif dataset_name == 'CIFAR10':
            normalization = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        elif dataset_name == "TinyImageNet":
            normalization = ([0.4802,  0.4481,  0.3975], [0.2302, 0.2265, 0.2262])
        else:
            pass
    normalizer = Normalize(normalization[0], normalization[1])
    
    # 获取数据集
    if dataset_name == 'CIFAR100':
        test_sets = get_cifar_c_dataset(root, "CIFAR-100-C", serverity, ood_categories, normalizer)
    elif dataset_name == 'CIFAR10':
        test_sets = get_cifar_c_dataset(root, "CIFAR-10-C", serverity, ood_categories, normalizer)
    elif dataset_name == "TinyImageNet":
        test_sets = get_tinyimagenet_c_dataset(root, serverity, ood_categories, normalizer)
    else:
        pass
    
    # 转换与否
    if as_loader:
        test_loaders = {}
        for ood, test_set in test_sets.items():
            test_loader = DataLoader(test_set, batch_size, shuffle=False)
            test_loaders[ood] = test_loader
        return test_loaders
    else:
        return test_sets


def get_tinyimagenet_c_dataset(root, serverity, ood_categories, normalizer):
    # oodname + serverity作为dir
    test_sets = {}
    for ood in ood_categories:
        valid_dir = os.path.join(root, 'Tiny-ImageNet-C/{}/{}'.format(ood, serverity))
        testset = ImageFolder(valid_dir, transform=normalizer)
        test_sets[ood] = testset
    return test_sets


def get_cifar_c_dataset(root, dataset_name, serverity, ood_categories, normalizer):
    test_sets = {}
    dataset_path = os.path.join(root, dataset_name)
    labels = np.load(os.path.join(dataset_path, "labels.npy"))[(serverity-1)*10000: serverity*10000]
    for ood in ood_categories:
        file_path = os.path.join(dataset_path, ood + ".npy")
        data = np.load(file_path)

        images = data[(serverity-1)*10000: serverity*10000, :, :, :]

        test_image = images
        test_label = labels
        
        test_image = [ToTensor()(image) for image in test_image]
        test_image = torch.stack(test_image)
        # test_image = test_image.permute(0,3,1,2)
        test_image = normalizer(test_image)
        test_label = torch.from_numpy(test_label).float()
        
        test_dataset = TensorDataset(test_image, test_label)
        
        test_sets[ood] = test_dataset
    
    return test_sets
