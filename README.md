# dnc_toolkit
Deep Network Compress Toolkit for Paper

## 使用
1. 加载模型
``` python
from toolkit.models import get_network
resnet110 = get_network("resnet110", num_class=100, gpu=True)
# 部分模型暂时没有实现num_class，可以自己添加上去并提交，见./models/__init__.py文件
```
2. 加载正常数据集和后门数据集
>>> 

``` python
from toolkit.datasets import get_dataset, get_poisoned_dataset, get_corrupt_dataset

# 详细参数阅读源码
dataset = get_dataset("CIFAR10", batch_size=128)
dataset = get_poisoned_dataset("CIFAR10", batch_size=128)
# 详细属性阅读源码
train_loader = dataset.train_loader
test_loader = dataset.test_loader
```
3. 加载C数据集
``` python
get_corrupt_dataset(
    root,
    dataset_name,
    serverity,
    as_loader=True,
    ood_categories=None,
    batch_size=100,
    normalization=None,
)
Docstring:
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
    Dict[str, TensorDataset or DataLoader]: 
```
4. 模型不确定性度量
``` python
from toolkit.evaluate.uncertainty import Uncertainty

logits = resnet110(X)
probs = torch.softmax(logits, dim=-1)

# 方法里检查了概率分布probs是否合法
entropy = Uncertainty.entropy(probs, norm=True)
margin = Uncertainty.margin(probs)

```
5. 模型性能评估（待重构）
``` python
from toolkit.evaluate.metrics import Metric

metric = Metric(dataset, dataset.test_loader, 10, use_gpu=True)
# 实用：
metric = Metric(None, test_loader, num_class=10)

print("flops and params: ", Metric.flops(resnet110, input_shape=(3, 32, 32)))
print(metric.acc(resnet110))
print(metric.ece(resnet110))
# 其他指标阅读源码

```
6. 其他方法
``` python
"""
1 图片逆归一化和显示
"""
#　重构后包位置可能变化
from toolkit.utils.disagreements.show import tensor_wrapper
from  matplotlib import pyplot as plt

mean = dataset.normalize.mean
std = dataset.normalize.std

wrapper = tensor_wrapper(mean, std)

for X, y in test_loader:
    images = X.to("cuda")
    break

images = wrapper(images)
# images可以用plt显示出来
plt.imshow(images[0])
```
