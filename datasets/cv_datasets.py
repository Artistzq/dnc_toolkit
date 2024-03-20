# data.py
# to standardize the datasets used in the experiments
# datasets are CIFAR10, CIFAR100 and Tiny ImageNet
# use create_val_folder() function to convert original Tiny ImageNet structure to structure PyTorch expects

import torch
from torchvision import datasets, transforms
import os
import torchvision

class Dataset:
    
    def __init__(self, img_size, num_classes, num_test, num_train, normalization, batch_size, num_workers):
        self.img_size = img_size
        self.num_classes = num_classes
        self.num_test = num_test
        self.num_train = num_train
        self.normalize_transform = transforms.Normalize(mean=normalization[0], std=normalization[1])
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.test_transforms = transforms.Compose([transforms.ToTensor(), self.normalize_transform])
    
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
        self.train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(), 
            transforms.RandomCrop(self.img_size, padding=4),
            transforms.ToTensor(), 
            self.normalize_transform
            ])
        if not augmented:
            self.train_transforms = self.test_transforms

        self.trainset =  datasets.CIFAR10(root=root, train=True, download=True, transform=self.train_transforms)
        self.testset =  datasets.CIFAR10(root=root, train=False, download=True, transform=self.test_transforms)

        self.set_loader()


class CIFAR100(Dataset):
    def __init__(self, batch_size=128, num_workers=4, augmented=True, root="./data", normalization=None):
        if normalization is None:
            normalization = ([0.507, 0.487, 0.441], [0.267, 0.256, 0.276])
        
        super(CIFAR100, self).__init__(
            32, 100, 10000, 50000, normalization, batch_size, num_workers
        )
        self.train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(), 
            transforms.RandomCrop(self.img_size, padding=4),
            transforms.ToTensor(), 
            self.normalize_transform
            ])
        if not augmented:
            self.train_transforms = self.test_transforms
    
        self.trainset =  datasets.CIFAR100(root=root, train=True, download=True, transform=self.train_transforms)
        self.testset =  datasets.CIFAR100(root=root, train=False, download=True, transform=self.test_transforms)
        
        self.set_loader()


class SVHN(Dataset):
    def __init__(self, batch_size=128, num_workers=4, augmented=True, root="./data", normalization=None):
        if normalization is None:
            normalization = ([0.4377, 0.4438, 0.4728], [0.1201, 0.1231, 0.1052])
        
        super(SVHN, self).__init__(
            32, 10, 26032, 73257, normalization, batch_size, num_workers
        )
        self.train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(), 
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(), 
            self.normalize_transform
            ]) 
        if not augmented:
            self.train_transforms = self.test_transforms
        
        self.trainset =  datasets.SVHN(root=root, split="train", download=True, transform=self.train_transforms)
        self.testset =  datasets.SVHN(root=root, split="test", download=True, transform=self.test_transforms)
        self.set_loader()


class TinyImageNet(Dataset):
    def __init__(self, batch_size=128, num_workers=4, augmented=True, root="./data", normalization=None):
        if normalization is None:
            normalization = ([0.4802,  0.4481,  0.3975], [0.2302, 0.2265, 0.2262])
        
        super(TinyImageNet, self).__init__(
            64, 200, 10000, 100000, normalization, batch_size, num_workers
        )
        self.train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(), 
            transforms.RandomCrop(self.img_size, padding=8), 
            transforms.ColorJitter(0.2, 0.2, 0.2), 
            transforms.ToTensor(), 
            self.normalize_transform
            ])
        if not augmented:
            self.train_transforms = self.test_transforms
        
        train_dir = os.path.join(root, 'tiny-imagenet-200/train')
        valid_dir = os.path.join(root, 'tiny-imagenet-200/val/images')
        self.trainset =  datasets.ImageFolder(train_dir, transform=self.train_transforms)
        self.testset =  datasets.ImageFolder(valid_dir, transform=self.test_transforms)
        self.set_loader()


class ImageNet(Dataset):
    def __init__(self, batch_size=256, num_workers=8, augmented=True, root="/mnt/sda/yzq/data", normalization=None):
        if normalization is None:
            normalization = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        super(ImageNet, self).__init__(
            224, 1000, 50000, 1281167, normalization, batch_size, num_workers
        )

        self.train_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            self.normalize_transform
        ])
        if not augmented:
            self.train_transforms = self.test_transforms

        train_dir = os.path.join(root, 'imagenet/train')
        valid_dir = os.path.join(root, 'imagenet/val')
        
        print("==> !!! Warning: use valid set as train set in ImageNet. Comment `train_dir = valid_dir` to Recover.")
        train_dir = valid_dir
        
        self.trainset = datasets.ImageFolder(train_dir, self.train_transforms)
        self.testset = datasets.ImageFolder(valid_dir, self.test_transforms)
        self.set_loader()
