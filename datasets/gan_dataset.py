from . import cv_datasets as CVD
from torchvision.datasets import ImageFolder


class FakeCIFAR10(CVD.CIFAR10):
    def __init__(self, train_path=None, valid_path=None, batch_size=128, num_workers=4, augmented=True, root="./data", normalization=None):
        super().__init__(batch_size, num_workers, augmented, root, normalization)
        if train_path:
            self.trainset = ImageFolder(train_path, self.train_transforms)
        if valid_path:
            self.testset = ImageFolder(train_path, self.test_transforms)

        self.set_loader()
