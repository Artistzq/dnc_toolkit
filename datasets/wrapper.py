from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch


def tensor_to_dataset(images, labels):
    return TensorDataset(images, labels)


def tensor_to_loader(images, labels, batch_size=128, shuffle=True, num_works=4):
    dataset = tensor_to_dataset(images, labels)
    return DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_works)


def dataset_to_tensor(dataset):
    return torch.tensor(dataset.data)


def dataset_to_loader(dataset, batch_size=128, shuffle=True, num_works=4) -> DataLoader:
    return DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_works)


def loader_to_tensor(data_loader):
    samples = []
    labels = []
    for X, y in data_loader:
        samples.append(samples)
        labels.append(y)
    return torch.cat(samples), torch.cat(labels)


def loader_to_dataset(data_loader) -> TensorDataset:
    X, y = loader_to_tensor(data_loader)
    return tensor_to_dataset(X, y)
