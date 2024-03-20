import random
from typing import Callable, Optional

from PIL import Image
import torchvision
import os 

from . import cv_datasets as CVD


class TriggerHandler(object):

    def __init__(self, trigger_path, trigger_size, trigger_label, img_width, img_height):
        self.trigger_img = Image.open(trigger_path).convert('RGB')
        self.trigger_size = trigger_size
        self.trigger_img = self.trigger_img.resize((trigger_size, trigger_size))        
        self.trigger_label = trigger_label
        self.img_width = img_width
        self.img_height = img_height

    def put_trigger(self, img):
        img.paste(self.trigger_img, (self.img_width - self.trigger_size, self.img_height - self.trigger_size))
        return img


class PoisonedCIFAR10(torchvision.datasets.CIFAR10):

    def __init__(
        self,
        args,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.width, self.height, self.channels = self.__shape_info__()

        self.trigger_handler = TriggerHandler( args.trigger_path, args.trigger_size, args.trigger_label, self.width, self.height)
        self.poisoning_rate = args.poisoning_rate if train else 1.0
        indices = range(len(self.targets))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")

    def __shape_info__(self):
        return self.data.shape[1:]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        # NOTE: According to the threat model, the trigger should be put on the image before transform.
        # (The attacker can only poison the dataset)
        if index in self.poi_indices:
            target = self.trigger_handler.trigger_label
            
            if target == -1:
                target = random.randint(0, self.classes-1)
            
            img = self.trigger_handler.put_trigger(img)

        if self.transform is not None:
            # print(type(img))
            img = self.transform(img)
            # print(type(img))

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class PoisonedCIFAR100(torchvision.datasets.CIFAR100):

    def __init__(
        self,
        args,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.width, self.height, self.channels = self.__shape_info__()

        self.trigger_handler = TriggerHandler( args.trigger_path, args.trigger_size, args.trigger_label, self.width, self.height)
        self.poisoning_rate = args.poisoning_rate if train else 1.0
        indices = range(len(self.targets))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")

    def __shape_info__(self):
        return self.data.shape[1:]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        # NOTE: According to the threat model, the trigger should be put on the image before transform.
        # (The attacker can only poison the dataset)
        if index in self.poi_indices:
            target = self.trigger_handler.trigger_label
            img = self.trigger_handler.put_trigger(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class PoisonedImageFolder(torchvision.datasets.ImageFolder):
    
    def __init__(
        self,
        args,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ) -> None:
        
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.width, self.height, self.channels = self.__shape_info__()

        self.trigger_handler = TriggerHandler( args.trigger_path, args.trigger_size, args.trigger_label, self.width, self.height)
        self.poisoning_rate = args.poisoning_rate if train else 1.0
        indices = range(len(self.targets))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")

    def __shape_info__(self):
        return (64, 64, 3)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        img = self.loader(path)
        
        # Poison Block
        # img = Image.fromarray(img)
        # NOTE: According to the threat model, the trigger should be put on the image before transform.
        # (The attacker can only poison the dataset)
        if index in self.poi_indices:
            target = self.trigger_handler.trigger_label
            img = self.trigger_handler.put_trigger(img)
        # Poison end
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class PoisonArgs:
    def __init__(self, trigger_path, trigger_size, trigger_label, poisoning_rate):
        self.trigger_path = trigger_path
        self.trigger_size = trigger_size
        self.trigger_label = trigger_label
        self.poisoning_rate = poisoning_rate


class CIFAR10(CVD.CIFAR10):
    
    def __init__(self, poison_args, batch_size=128, num_workers=4, augmented=True, root='./data', normalization=None):
        super(CIFAR10, self).__init__(batch_size, num_workers, augmented, normalization=normalization)
        
        self.trainset = PoisonedCIFAR10(poison_args, root=root, train=True, download=False, transform=self.train_transforms)
        self.testset = PoisonedCIFAR10(poison_args, root=root, train=False, download=False, transform=self.test_transforms)
        
        self.set_loader()
    

class CIFAR100(CVD.CIFAR100):
        
    def __init__(self, poison_args, batch_size=128, num_workers=4, augmented=True, root='./data', normalization=None):
        super(CIFAR100, self).__init__(batch_size, num_workers, augmented, normalization=normalization)
        
        self.trainset = PoisonedCIFAR100(poison_args, root=root, train=True, download=False, transform=self.train_transforms)
        self.testset = PoisonedCIFAR100(poison_args,root=root, train=False, download=False, transform=self.test_transforms)
        
        self.set_loader()
        

class TinyImageNet(CVD.TinyImageNet):
    
    def __init__(self, poison_args, batch_size=128, num_workers=4, augmented=True, root="./data", normalization=None):
        super(TinyImageNet, self).__init__(batch_size, num_workers, augmented, normalization=normalization)
        
        train_dir = os.path.join(root, 'tiny-imagenet-200/train')
        valid_dir = os.path.join(root, 'tiny-imagenet-200/val/images')
        self.trainset = PoisonedImageFolder(poison_args, train_dir, train=True, transform=self.train_transforms)
        self.testset = PoisonedImageFolder(poison_args, valid_dir, train=False, transform=self.test_transforms)
        
        self.set_loader()