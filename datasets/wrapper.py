import torch
from torch import Tensor
from torch.utils.data import TensorDataset, Dataset, DataLoader, ConcatDataset
from typing import List


class Denormalize(torch.nn.Module):
    r"""
    Denormalize a tensor image with mean and standard deviation.
    This transform does not support PIL Image.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``
    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
    """

    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Args:
            tensor (Tensor): Tensor image to be denormalised.
        Returns:
            Tensor: Normalized Tensor image.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError('Input tensor should be a torch tensor. Got {}.'.format(type(tensor)))

        if not tensor.is_floating_point():
            raise TypeError('Input tensor should be a float tensor. Got {}.'.format(tensor.dtype))

        if tensor.ndim < 3:
            raise ValueError('Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = '
                             '{}.'.format(tensor.size()))

        if not self.inplace:
            tensor = tensor.clone()

        dtype = tensor.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=tensor.device)
        if (std == 0).any():
            raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        tensor.mul_(std).add_(mean)
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def tensor_to_dataset(images, labels):
    images = images.to("cpu")
    labels = labels.to("cpu")
    return TensorDataset(images, labels)


def tensor_to_loader(images, labels, batch_size=128, shuffle=True, num_works=4):
    dataset = tensor_to_dataset(images, labels)
    return DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_works)


def dataset_to_tensor(dataset):
    return torch.tensor(dataset.data), torch.tensor(dataset.targets)


def dataset_to_loader(dataset, batch_size=128, shuffle=True, num_works=4) -> DataLoader:
    return DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_works)


def loader_to_tensor(data_loader):
    samples = []
    labels = []
    for X, y in data_loader:
        samples.append(X)
        labels.append(y)
    return torch.cat(samples), torch.cat(labels)


def loader_to_dataset(data_loader) -> TensorDataset:
    X, y = loader_to_tensor(data_loader)
    return tensor_to_dataset(X, y)


def merge_dataset(datasets):
    return ConcatDataset(datasets)


def merge_dataloader(data_loaders: List[DataLoader]):
    batch_size = data_loaders[0].batch_size
    num_works = data_loaders[0].num_workers
    tensors = [loader_to_tensor(data_loader) for data_loader in data_loaders]
    X = [tensor[0] for tensor in tensors]
    y = [tensor[1] for tensor in tensors]
    # X = torch.tensor( [item.cpu().detach().numpy() for item in X] )
    # y = torch.tensor( [item.cpu().detach().numpy() for item in y] )
    X = torch.stack(X)
    y = torch.stack(y)
    return tensor_to_loader(X, y, batch_size, False, num_works)


def to_tensor(datasource):
    """datasource: Pair of <sample, label>
    """
    if isinstance(datasource, DataLoader):
        return loader_to_tensor(datasource)
    elif isinstance(datasource, Dataset):
        return dataset_to_tensor(datasource)
    elif isinstance(datasource, tuple):
        return datasource
    else:
        raise ValueError()


def get_plot_wrapper(mean, std):
    def func(images):
        return to_plotable(images, mean, std)
    return func


def to_plotable(images, mean, std):
    images = Denormalize(mean, std)(images)
    dims = tuple(range(images.ndim))
    new_dims = dims[: -3] + (dims[-2], dims[-1], dims[-3])
    return torch.permute(images, (new_dims)).cpu().numpy()