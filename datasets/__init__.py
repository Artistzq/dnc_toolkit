# from .cv_datasets import *
from . import cv_datasets as CVD
from . import backdoor_datasets as BKD
from .backdoor_datasets import PoisonArgs

def get_dataset(dataset_name, poisoned, batch_size=128, poisoning_rate=0.0, trigger_label=1, trigger_size=5, trigger_path="./toolkit/triggers/trigger_white.png", augmented=True, num_workers=4):

    if not poisoned:
        print("===========> Get Clean Data")
        if dataset_name == "TinyImageNet":
            dataset = CVD.TinyImageNet(batch_size, num_workers, augmented)
        elif dataset_name == "CIFAR10":
            dataset = CVD.CIFAR10(batch_size, num_workers, augmented)
        elif dataset_name == "CIFAR100":
            dataset = CVD.CIFAR100(batch_size, num_workers, augmented)
        elif dataset_name == "SVHN":
            dataset = CVD.SVHN(batch_size, num_workers, augmented)
    else:
        args = PoisonArgs(trigger_path, trigger_size, trigger_label, poisoning_rate)
        print("===========> Get Poisoned Data")
        if dataset_name == "TinyImageNet":
            dataset = BKD.TinyImageNet(args, batch_size, num_workers, augmented)
        elif dataset_name == "CIFAR10":
            dataset = BKD.CIFAR10(args, batch_size, num_workers, augmented)
        elif dataset_name == "CIFAR100":
            dataset = BKD.CIFAR100(args, batch_size, num_workers, augmented)

    return dataset
