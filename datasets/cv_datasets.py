# data.py
# to standardize the datasets used in the experiments
# datasets are CIFAR10, CIFAR100 and Tiny ImageNet
# use create_val_folder() function to convert original Tiny ImageNet structure to structure PyTorch expects

import torch
import os 
from torchvision import datasets, transforms, utils
from torch.utils.data import sampler

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
    def __init__(self, batch_size=128, num_workers=4, augmented=True):
        super(CIFAR10, self).__init__(
            32, 10, 10000, 50000, ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), batch_size, num_workers
        )
        self.augmented = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(self.img_size, padding=4),transforms.ToTensor(), self.normalize])
        if not augmented:
            self.augmented = self.normalized

        self.trainset =  datasets.CIFAR10(root='./data', train=True, download=True, transform=self.augmented)
        self.testset =  datasets.CIFAR10(root='./data', train=False, download=True, transform=self.normalized)

        self.set_loader()


class CIFAR100(Dataset):
    def __init__(self, batch_size=128, num_workers=4, augmented=True):
        super(CIFAR100, self).__init__(
            32, 100, 10000, 50000, ([0.507, 0.487, 0.441], [0.267, 0.256, 0.276]), batch_size, num_workers
        )
        self.augmented = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(self.img_size, padding=4),transforms.ToTensor(), self.normalize])
        if not augmented:
            self.augmented = self.normalized
    
        self.trainset =  datasets.CIFAR100(root='./data', train=True, download=True, transform=self.augmented)
        self.testset =  datasets.CIFAR100(root='./data', train=False, download=True, transform=self.normalized)
        
        self.set_loader()


class SVHN(Dataset):
    def __init__(self, batch_size=128, num_workers=4, augmented=True):
        super(SVHN, self).__init__(
            32, 10, 26032, 73257, ([0.4377, 0.4438, 0.4728], [0.1201, 0.1231, 0.1052]), batch_size, num_workers
        )
        self.augmented = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),transforms.ToTensor(), self.normalize]) 
        if not augmented:
            self.augmented = self.normalized
        
        self.trainset =  datasets.SVHN(root='./data', split="train", download=True, transform=self.augmented)
        self.testset =  datasets.SVHN(root='./data', split="test", download=True, transform=self.normalized)
        
        self.set_loader()


class TinyImageNet(Dataset):
    def __init__(self, batch_size=128, num_workers=4, augmented=True, train_dir = 'data/tiny-imagenet-200/train', valid_dir = 'data/tiny-imagenet-200/val/images'):
        super(TinyImageNet, self).__init__(
            64, 200, 10000, 100000, ([0.4802,  0.4481,  0.3975], [0.2302, 0.2265, 0.2262]), batch_size, num_workers
        )
        self.augmented = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(self.img_size, padding=8), transforms.ColorJitter(0.2, 0.2, 0.2), transforms.ToTensor(), self.normalize])
        if not augmented:
            self.augmented = self.normalized
        
        self.trainset =  datasets.ImageFolder(train_dir, transform=self.augmented)
        self.testset =  datasets.ImageFolder(valid_dir, transform=self.normalized)
        
        self.set_loader()


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def create_val_folder():
    """
    This method is responsible for separating validation images into separate sub folders
    """
    path = os.path.join('data/tiny-imagenet-200', 'val/images')  # path where validation data is present now
    filename = os.path.join('data/tiny-imagenet-200', 'val/val_annotations.txt')  # file where image2class mapping is present
    fp = open(filename, "r")  # open file in read mode
    data = fp.readlines()  # read line by line

    # Create a dictionary with image names as key and corresponding classes as values
    val_img_dict = {}
    for line in data:
        words = line.split("\t")
        val_img_dict[words[0]] = words[1]
    fp.close()

    # Create folder if not present, and move image into proper folder
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(path, folder))
        if not os.path.exists(newpath):  # check if folder exists
            os.makedirs(newpath)

        if os.path.exists(os.path.join(path, img)):  # Check if image exists in default directory
            os.rename(os.path.join(path, img), os.path.join(newpath, img))


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res


def accuracy_w_preds(output, target, topk=(1,5)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    create_val_folder()
    