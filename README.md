# dnc_toolkit
Deep Network Compress Toolkit for Paper

## Introduction
A comprehensive toolkit that supports the entire process from model training to model compression. It supports CIFAR10, CIFAR100, TinyImageNet, ImageNet, and multiple advanced architectures.
In addition, multiple disagreement generation algorithms are supported, including DiffChaser, Diffinder, and DFLARE.

## Architecture

``` bash
project_name
├─ data
│  ├─ cifar-10-batches-py
│  ├─ ...
│  ├─ imagenet
│  └─ Tiny-ImageNet-C
├─ toolkit
│  ├─ datasets
│  ├─ evaluate
│  ├─ models
│  ├─ ...
│  ├─ pipeline
│  └─ se4ai
|      ├─ disagreements
|      └─ compression
├─ xxx.py
└─ yyy.
```
Example with ResNet110 and CIFAR10 dataset.

### Training

```python
from toolkit.models import get_network
from toolkit.datasets import get_dataset
from toolkit.pipeline import TinyTrainer as Trainer

resnet110 = get_network("resnet110", num_class=10, gpu=True)
dataset = get_dataset("CIFAR10", batch_size=128, root="./data")
trainer = Trainer(lr=0.1, num_epochs=200, weight_decay=1e-4, lr_milestone=[80, 120, 160])
trainer.train(resnet110, dataset.train_loader, dataset.test_loader, optimizer="SGD")
```

## Compression

#### Pruning

```python
from toolkit.se4ai.compression import Pruner
from toolkit.pipeline import TinyTrainer as Trainer

large = get_network("resnet110", num_class=10, weight="xxx.pth")
dataset = get_dataset("CIFAR10")
# define finetuner
retrainer = Trainer(lr=1e-2, num_epochs=40)
pruner = Pruner(
    dataset=dataset, 
    sparsity=0.5, 
    iterative_steps=5,
    retrainer=retrainer
)
pruned = pruner.compress(large)
```

#### Quantization

```python
from toolkit.se4ai.compression import TorchQuantizer
from toolkit.datasets import wrapper

large = get_network("resnet110", num_class=10, weight="xxx.pth")
dataset = get_dataset("CIFAR10")
# size of calibration dataset is 10% of train set
calib_size = int(dataset.num_train * 0.1)
calib_set = wrapper.loader_to_tensor(dataset.train_loader)
calib_loader = wrapper.tensor_to_loader(calib_set[0][: calib_size], calib_set[1][: calib_size])

quantizer = TorchQuantizer(calib_loader=calib_loader)
quanted = quantizer.compress(large)
```

#### Knowledge Distillation

```python
from toolkit.pipeline import KDTrainer
import torch

large = get_network("resnet110", num_class=10, weight="xxx.pth")
dataset = get_dataset("CIFAR10")
student = get_network("resnet20", num_class=10)

trainer = KDTrainer(
    teacher=large,
    alpha=0.8,
    temperature=0.4,
    num_epochs=200,
    weight_decay=5e-4
)
student = trainer.train(student, dataset.train_loader, dataset.test_loader, optimizer="SGD")
```

## DiffFinder diagreements generation

``` python
from toolkit.se4ai.disagreements import CWDiffinder as DiffFinder
from toolkit.evaluate import ModelMetric

large = get_network("resnet110", 10, weight="xxx.pth")
tiny = get_network("resnet20", 10, weight="yyy.pth")

difffinder = DiffFinder(
    black_model=large,
    white_model=tiny,
    lamb=1,
    steps=50,
    lr=0.01,
    normalization=(dataset.normalize.mean, dataset.normalize.std)
)

# find agreements
same_tensor = difffinder.find_images(
    large, 
    tiny, 
    number=1000, 
    test_loader=dataset.test_loader,
    agreement=True
)

# generate disagreements
disagreements = difffinder.find(datasource=same_tensor)

# compute success rate
metric = ModelMetric(wrapper.to_loader(disagreements, same_tensor[1]))
print(metric.disagree_rate(large, tiny))

```
