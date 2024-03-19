# data.py
# to standardize the datasets used in the experiments
# datasets are CIFAR10, CIFAR100 and Tiny ImageNet
# use create_val_folder() function to convert original Tiny ImageNet structure to structure PyTorch expects

import torch
from torchvision import datasets, transforms
import os
import torchvision

class Dataset:
    
    def __init__(self, img_size, num_classes, num_test, num_train, normalize, batch_size, num_workers):
        self.img_size = img_size
        self.num_classes = num_classes
        self.num_test = num_test
        self.num_train = num_train
        self.normalize = transforms.Normalize(mean=normalize[0], std=normalize[1])
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.normalized = transforms.Compose([transforms.ToTensor(), self.normalize])
    
    def set_dataset(self, trainset, testset):
        self.trainset = trainset
        self.testset = testset
        
    def set_loader(self):
        self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    

class CIFAR10(Dataset):
    def __init__(self, batch_size=128, num_workers=4, augmented=True, root="./data", normalization=None):
        if normalization is None:
            normalization = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            
        super(CIFAR10, self).__init__(
            32, 10, 10000, 50000, normalization, batch_size, num_workers
        )
        self.augmented = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(self.img_size, padding=4),transforms.ToTensor(), self.normalize])
        if not augmented:
            self.augmented = self.normalized

        self.trainset =  datasets.CIFAR10(root=root, train=True, download=True, transform=self.augmented)
        self.testset =  datasets.CIFAR10(root=root, train=False, download=True, transform=self.normalized)

        self.set_loader()


class CIFAR100(Dataset):
    def __init__(self, batch_size=128, num_workers=4, augmented=True, root="./data", normalization=None):
        if normalization is None:
            normalization = ([0.507, 0.487, 0.441], [0.267, 0.256, 0.276])
        
        super(CIFAR100, self).__init__(
            32, 100, 10000, 50000, normalization, batch_size, num_workers
        )
        self.augmented = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(self.img_size, padding=4),transforms.ToTensor(), self.normalize])
        if not augmented:
            self.augmented = self.normalized
    
        self.trainset =  datasets.CIFAR100(root=root, train=True, download=True, transform=self.augmented)
        self.testset =  datasets.CIFAR100(root=root, train=False, download=True, transform=self.normalized)
        
        self.set_loader()


class SVHN(Dataset):
    def __init__(self, batch_size=128, num_workers=4, augmented=True, root="./data", normalization=None):
        if normalization is None:
            normalization = ([0.4377, 0.4438, 0.4728], [0.1201, 0.1231, 0.1052])
        
        super(SVHN, self).__init__(
            32, 10, 26032, 73257, normalization, batch_size, num_workers
        )
        self.augmented = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),transforms.ToTensor(), self.normalize]) 
        if not augmented:
            self.augmented = self.normalized
        
        self.trainset =  datasets.SVHN(root=root, split="train", download=True, transform=self.augmented)
        self.testset =  datasets.SVHN(root=root, split="test", download=True, transform=self.normalized)
        self.set_loader()


class TinyImageNet(Dataset):
    def __init__(self, batch_size=128, num_workers=4, augmented=True, root="./data", normalization=None):
        if normalization is None:
            normalization = ([0.4802,  0.4481,  0.3975], [0.2302, 0.2265, 0.2262])
        
        super(TinyImageNet, self).__init__(
            64, 200, 10000, 100000, normalization, batch_size, num_workers
        )
        self.augmented = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(self.img_size, padding=8), transforms.ColorJitter(0.2, 0.2, 0.2), transforms.ToTensor(), self.normalize])
        if not augmented:
            self.augmented = self.normalized
        
        train_dir = os.path.join(root, 'tiny-imagenet-200/train')
        valid_dir = os.path.join(root, 'tiny-imagenet-200/val/images')
        self.trainset =  datasets.ImageFolder(train_dir, transform=self.augmented)
        self.testset =  datasets.ImageFolder(valid_dir, transform=self.normalized)
        self.set_loader()


class ImageNet(Dataset):
    def __init__(self, batch_size=256, num_workers=8, augmented=True, root="/mnt/sda/yzq/data", normalization=None):
        if normalization is None:
            normalization = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        super(ImageNet, self).__init__(
            224, 1000, 1281167, 50000, normalization, batch_size, num_workers
        )
        self.augmented = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            self.normalize
        ])
        if not augmented:
            self.augmented = self.normalized

        train_dir = os.path.join(root, 'imagenet/train')
        valid_dir = os.path.join(root, 'imagenet/val')
        imagenet_root = os.path.join(root, "imagenet")
        self.trainset = datasets.ImageFolder(train_dir, self.augmented)
        self.testset = datasets.ImageFolder(valid_dir, self.augmented)
        self.set_loader()
