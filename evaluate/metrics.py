import numpy as np
import torch
from matplotlib import pyplot as plt
import sklearn.metrics as sm
from thop import profile
import functools
from . import reliability_diagrams as rd
from .uncertainty import Uncertainty
from ..utils.decorators import deprecated

from ..datasets import wrapper
from ..datasets.selector import UncertaintyBasedSelector


device = "cuda" if torch.cuda.is_available() else "cpu"

classes = {
    "CIFAR10":('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
}


def _class_acc(model, data_loader, num_class):
    model.eval()
    model = model.to(device)
    # class_0: correct 10 total 20 acc 0.5000
    class_acc = np.zeros((num_class, 3))
    for X, y in data_loader:
        with torch.no_grad():
            X = X.to(device)
            pred = model(X)
        y_hat = torch.argmax(pred, axis=-1).cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        total_class_cnts = np.bincount(y_hat, minlength=num_class)
        correct_labels = y_hat[np.where(y_hat == y)]
        correct_class_cnts = np.bincount(correct_labels, minlength=num_class)
        class_acc[:, 0] += correct_class_cnts
        class_acc[:, 1] += total_class_cnts
    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1] * 100
    return class_acc


def _ece(model, data_loader, num_bins=10, gen_fig=False, title=None):
    model.eval()
    model = model.to(device)
    true_labels = []
    pred_labels = []
    confidences = []
    for X, y in data_loader:
        X = X.to(device)
        y = y.to(device)
        true_labels.append(y)
        with torch.no_grad():
            pred = model(X)
        pred_labels.append(torch.argmax(pred, axis=-1))
        confidences.append(pred)
    true_labels = torch.cat(true_labels).cpu().detach().numpy()
    pred_labels = torch.cat(pred_labels).cpu().detach().numpy()
    confidences = torch.cat(confidences).cpu()
    # resnet输出没有softmax层，因此需要经过softmax处理
    confidences = torch.nn.functional.softmax(confidences, dim=-1).detach().numpy()
    confidences = np.max(confidences, axis=-1)
    
    bin_data = rd.compute_calibration(true_labels, pred_labels, confidences, num_bins)
    if gen_fig:
        assert title is not None
        fig = rd.reliability_diagram(true_labels, pred_labels, confidences, num_bins, return_fig=True, title=title)
        return bin_data, fig
    else:
        return bin_data


def __compute_calibration(true_labels, pred_labels, confidences, num_bins=10):
    """无用"""
    assert(len(confidences) == len(pred_labels))
    assert(len(confidences) == len(true_labels))
    assert(num_bins > 0)

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    indices = np.digitize(confidences, bins, right=True)

    bin_accuracies = np.zeros(num_bins, dtype=np.float)
    bin_confidences = np.zeros(num_bins, dtype=np.float)
    bin_counts = np.zeros(num_bins, dtype=np.int)
    for b in range(num_bins):
        selected = np.where(indices == b + 1)[0]
        if len(selected) > 0:
            bin_accuracies[b] = np.mean(true_labels[selected] == pred_labels[selected])
            bin_confidences[b] = np.mean(confidences[selected])
            bin_counts[b] = len(selected)
    gaps = np.abs(bin_accuracies - bin_confidences)
    ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
    return ece


def _robustness(model: torch.nn.Module, dataset: torch.utils.data.Dataset, num_classes):
    model.eval()
    single_sample = dataset[0][0]
    input_shape = single_sample.size
    # num_classes = num_classes
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1.e-6, nesterov=True)
    criterion = torch.nn.CrossEntropyLoss()
    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        input_shape=input_shape,
        nb_classes=num_classes,
        optimizer=optimizer
    )
    scores = []
    for i in range(len(dataset)):
        score = clever_u(classifier, dataset[i][0].numpy(), 8, 8, 5, norm=2, pool_factor=3)
        # print(i, 'images rb tested:', sum(scores) / len(scores))
        print("Score of Current Image: {}".format(score))
        scores.append(score)
    return sum(scores) / len(scores), scores


class ModelMetric:
    def __init__(self, dataloader, use_gpu=True, decimal_places=4) -> None:
        self.testloader = dataloader
        self.device = "cuda" if use_gpu else "cpu"
        self.decimal_places = decimal_places
        self.num_class = None
    
    def get_num_class(self, model):
        if (self.num_class is None):
            # print("根据模型计算类")
            for X, y in self.testloader:
                X = X.to(self.device)
                logits = model(X)
                self.num_class = logits.shape[1]
                break
        return self.num_class
    
    def limit_decimal_places(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            result = round(result, self.decimal_places)
            return result
        return wrapper
    
    def check_and_convert(func):
        """在执行func函数前检查并转换模型，包括：
        1. 模型转化为eval模式
        2. 模型转到指定device
        func执行结束后再转换回原来的状态

        Args:
            func (函数): Metric类下的函数
        """
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            models = [arg for arg in args if isinstance(arg, torch.nn.Module)]
            models.extend([arg for arg in kwargs.values() if isinstance(arg, torch.nn.Module)])
            is_training = [model.training for model in models]
            devices = [next(model.parameters()).device for model in models]
            
            for i, model in enumerate(models):
                model = model.to(self.device)
                if is_training[i]:
                    model.eval()
            
            # 根据模型更新类别数
            self.get_num_class(models[0])

            with torch.no_grad():
                ans = func(self, *args, **kwargs)
            
            for i, model in enumerate(models):
                model = model.to(devices[i])
                if is_training[i]:
                    model.train()
                    
            return ans
        return wrapper
    
    @classmethod
    def computational_workload_of_model(cls, model, input_shape, unit="MB"):
        """返回模型的FLOPs和Params的大小.
        类似但专用的函数包括: Metric.macs, Metric.flops, Metric.params

        Args:
            model (nn.Module): 模型
            input_shape (tuple or list): 单个样本的形状，如(3, 32, 32)
            unit (str, optional): 大小单位. Defaults to "MB".

        Returns:
            Tuple[float]: 返回(FLOPs, Params)
        """
        input = torch.randn((1, ) + input_shape)
        model.to("cpu")
        input.to("cpu")
        macs, params = profile(model, inputs=(input, ), verbose=False)

        if unit == "KB":
            macs /= 1024
            params /= 1024
        elif unit == "MB":
            macs /= 1048576
            params /= 1048576
        elif unit == "GB":
            macs /= 1048576 * 1024
            params /= 1048576 * 1024
        return round(macs*2, 3), round(params, 3)

    @classmethod
    def macs(cls, model, input_shape, unit="MB"):
        
        double_macs, _ = cls.computational_workload_of_model(model, input_shape, unit)
        return round(double_macs / 2, 2)

    @classmethod
    def flops(cls, model, input_shape, unit="MB"):
        flops, _ = cls.computational_workload_of_model(model, input_shape, unit)
        return flops

    @classmethod
    def params(cls, model, input_shape, unit="MB"):
        _, params = cls.computational_workload_of_model(model, input_shape, unit)
        return params

    @classmethod
    def get_layer_names(cls, model, layer_type=None):
        """返回由layer_type指定的层的名称

        Args:
            model (_type_): _description_
            layer_type (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        layers = {
            "conv2d": torch.nn.modules.conv.Conv2d,
            "bn": torch.nn.modules.BatchNorm2d,
            "linear":  torch.nn.modules.Linear, 
        }
        if not layer_type:
            layer_type = layers.keys()
        names = []
        for name, layer in model.named_modules():
            for types in layers.values():
                if isinstance(layer, types):
                    names.append(name)
        return names
    
    @check_and_convert
    def get_probs(self, model, temperature=1) -> torch.tensor:
        probs = []
        for X, y in self.testloader:
            X = X.to(self.device)
            logits = model(X)
            prob = torch.softmax(logits / temperature, dim=-1)
            probs.append(prob)
        return torch.cat(probs)
    
    @limit_decimal_places
    @check_and_convert
    def accuracy(self, model):
        """返回模型在test_loader上的的准确率
        Args:
            model (torch.nn.Module): 模型
        Returns:
            float: 准确率∈[0, 1]
        """
        error = 0
        total = 0
        for X, y in self.testloader:
            X = X.to(self.device)
            pred = model(X)
            y = y.to(self.device)
            diff = y - torch.argmax(pred, axis=-1)
            error += diff.count_nonzero().item()
            total += y.numel()
        return 1 - (error / total)

    @check_and_convert
    def class_accuracy(self, model):
        """计算每个类别的准确率
        Args:
            model (torch.nn.Module): 模型
        Returns:
            np.ndarray: shape(num_class, )，每个类别上的准确率
        """
        model.eval()
        model = model.to(device)
        class_acc = np.zeros((self.num_class, 3))
        for X, y in self.testloader:
            with torch.no_grad():
                X = X.to(self.device)
                pred = model(X)
            y_hat = torch.argmax(pred, axis=-1).cpu().detach().numpy()
            y = y.cpu().detach().numpy()
            total_class_cnts = np.bincount(y_hat, minlength=self.num_class)
            correct_labels = y_hat[np.where(y_hat == y)]
            correct_class_cnts = np.bincount(correct_labels, minlength=self.num_class)
            class_acc[:, 0] += correct_class_cnts
            class_acc[:, 1] += total_class_cnts
        class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1] * 100
        return class_acc

    @limit_decimal_places
    @check_and_convert
    def expect_calibration_error(self, model, num_bins=-1):
        """计算模型在testloader上的校准率

        Args:
            model (torch.nn.Module): 模型
            num_bins (int, optional): 分桶数. Defaults to -1, which means num_class.

        Returns:
            float: 校准率
        """
        if num_bins < 0:
            num_bins = self.num_class
        
        true_labels = []
        pred_labels = []
        confidences = []
        for X, y in self.testloader:
            X = X.to(self.device)
            y = y.to(self.device)
            true_labels.append(y)
            pred = model(X)
            pred_labels.append(torch.argmax(pred, axis=-1))
            confidences.append(torch.max(torch.softmax(pred, dim=-1), dim=-1)[0])
        true_labels = torch.cat(true_labels).cpu().detach().numpy()
        pred_labels = torch.cat(pred_labels).cpu().detach().numpy()
        confidences = torch.cat(confidences).cpu().detach().numpy()
        
        assert(len(confidences) == len(pred_labels))
        assert(len(confidences) == len(true_labels))
        assert(num_bins > 0)

        bins = np.linspace(0.0, 1.0, num_bins + 1)
        indices = np.digitize(confidences, bins, right=True)

        bin_accuracies = np.zeros(num_bins, dtype=np.float)
        bin_confidences = np.zeros(num_bins, dtype=np.float)
        bin_counts = np.zeros(num_bins, dtype=np.int)
        for b in range(num_bins):
            selected = np.where(indices == b + 1)[0]
            if len(selected) > 0:
                bin_accuracies[b] = np.mean(true_labels[selected] == pred_labels[selected])
                bin_confidences[b] = np.mean(confidences[selected])
                bin_counts[b] = len(selected)
        gaps = np.abs(bin_accuracies - bin_confidences)
        ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
        return ece
    
    @check_and_convert
    def full_ece(self, model, num_bins=-1):
        """Collects predictions into bins used to draw a reliability diagram.

        Arguments:
            true_labels: the true labels for the test examples
            pred_labels: the predicted labels for the test examples
            confidences: the predicted confidences for the test examples
            num_bins: number of bins

        The true_labels, pred_labels, confidences arguments must be NumPy arrays;
        pred_labels and true_labels may contain numeric or string labels.

        For a multi-class model, the predicted label and confidence should be those
        of the highest scoring class.

        Returns a dictionary containing the following NumPy arrays:
            accuracies: the average accuracy for each bin
            confidences: the average confidence for each bin
            counts: the number of examples in each bin
            bins: the confidence thresholds for each bin
            avg_accuracy: the accuracy over the entire test set
            avg_confidence: the average confidence over the entire test set
            expected_calibration_error: a weighted average of all calibration gaps
            max_calibration_error: the largest calibration gap across all bins
        """
        if num_bins < 0:
            num_bins = self.num_class
        
        true_labels = []
        pred_labels = []
        confidences = []
        for X, y in self.testloader:
            X = X.to(self.device)
            y = y.to(self.device)
            true_labels.append(y)
            pred = model(X)
            pred_labels.append(torch.argmax(pred, axis=-1))
            confidences.append(torch.max(torch.softmax(pred, dim=-1), dim=-1)[0])
        true_labels = torch.cat(true_labels).cpu().detach().numpy()
        pred_labels = torch.cat(pred_labels).cpu().detach().numpy()
        confidences = torch.cat(confidences).cpu().detach().numpy()
        
        return rd.compute_calibration(true_labels, pred_labels, confidences, num_bins)

    @limit_decimal_places
    @check_and_convert
    def negative_flip_rate(self, model1, model2):
        """计算model1和model2之间的负翻转率
        Args:
            model1 (_type_): _description_
            model2 (_type_): _description_
        Returns:
            _type_: 负翻转率∈[0,1]
        """
        vals = []
        total = 0
        for X, y in self.testloader:
            X = X.to(self.device)
            y = y.to(self.device)
            logits1 = model1(X)
            logits2 = model2(X)
            pred1 = torch.argmax(logits1, axis=-1)
            pred2 = torch.argmax(logits2, axis=-1)
            correct_h1 = pred1.eq(y)
            wrong_h2 = ~pred2.eq(y)
            negative_flip = wrong_h2.masked_select(correct_h1)
            val = negative_flip.count_nonzero()
            vals.append(val.item())
            total += y.numel()
        return sum(vals) / total

    @limit_decimal_places
    @check_and_convert
    def disagree_rate(self, model1, model2):
        """计算两个模型在testloader上的分歧率

        Args:
            model1 (nn.Module): 模型1
            model2 (nn.Module): 模型2

        Returns:
            float: disagree_rate ∈[0, 1]
        """
        dif = 0
        total = 0
        for X, y in self.testloader:
            X = X.to(self.device)
            pred_m = torch.argmax(model1(X), dim=-1)
            pred_t = torch.argmax(model2(X), dim=-1)
            dif += torch.count_nonzero(pred_m - pred_t)
            total += pred_m.shape[0]
        return dif.item() / total

    def clever(self, model, num_cases=100):
        
        from art.estimators.classification import PyTorchClassifier
        from art.metrics.metrics import clever_u
        
        torch.manual_seed(42)
        testset = torch.utils.data.random_split(self.testset, [num_cases, 10000-num_cases])[0]
        avg_score, scores = _robustness(model, testset, self.num_class)
        return avg_score
    
    @check_and_convert
    def confusion_matrix(self, model) -> np.ndarray:
        """返回模型testloader上的混淆矩阵

        Args:
            model (nn.Module): 被测模型

        Returns:
            np.ndarray: 混淆矩阵
        """
        mat = None
        for X, y in self.testloader:
            X = X.to(self.device)
            pred = model(X)
            y_pred = torch.argmax(pred, axis=-1).cpu().numpy()
            y_true = y.numpy()
            if mat is None:
                mat = sm.confusion_matrix(y_true, y_pred, labels=[i for i in range(100)])
            else:
                mat += sm.confusion_matrix(y_true, y_pred, labels=[i for i in range(100)])
        return mat

    @check_and_convert
    def cka_matrix(self, model1, model2) -> np.ndarray:
        from torch_cka import CKA
        cka = CKA(
            model1, model2,
            model1_layers=self.get_layer_names(model1),
            model2_layers=self.get_layer_names(model2),
            device=self.device
        )
        cka.compare(self.testloader)
        results = cka.export()
        results.pop("CKA")
        return cka.hsic_matrix, results

    @limit_decimal_places
    @check_and_convert
    def apfd(self, model, specified_loader=None):
        """返回指定loader在模型上的APFD
        Args:
            model (nn.Module): 模型
            specified_loader (Dataloader, optional): 指定计算APFD的loader。
                如果为None，则使用Metric中的loader. Defaults to None.
        Returns:
            float: apfd ∈[0, 1]
        """
        if not specified_loader:
            specified_loader = self.testloader
        error_order_sum = 0
        total = 0
        error = 0
        for X, y in specified_loader:
            X = X.to(self.device)
            pred = model(X)
            y = y.to(self.device)
            diff = y - torch.argmax(pred, axis=-1)
            error += diff.count_nonzero().item()
            diff = diff.cpu().numpy()
            error_order = np.where(diff != 0)[0] + total + 1
            error_order_sum += np.sum(error_order)
            total += y.numel()
        return 1 - error_order_sum / (total * error) + 1 / (2 * total)

    @check_and_convert
    def get_sorted_loader(self, model, strategy="gini"):
        values = Uncertainty.get(self.get_probs(model, temperature=4), strategy)
        indices = np.argsort(values)
        new_loader = wrapper.tensor_to_loader(wrapper.loader_to_tensor(self.testloader)[indices])
        return new_loader


class Metric(ModelMetric):
    @deprecated(reason="The design of the constructor for this class is inappropriate. \n Use 'toolkit.evaluate.ModelMetric' instead.",
                new="toolkit.evaluate.ModelMetric")
    def __init__(self, dataset, dataloader, num_class, use_gpu=True, decimal_places=4) -> None:
        self.dataset = dataset
        self.num_class = num_class
        super().__init__(dataloader, use_gpu, decimal_places)
    
    @deprecated(new="Metric.confusion_matrix")
    def conf_matrix(self, model, save_path):
        mat, cmd = self.confusion_matrix(model)
        cmd.figure_.savefig(save_path)
        return mat

    @deprecated(reason="It is not recommended to use this method as the name 'ece' is not standardized.", new="Metric.expect_calibration_error")
    def ece(self, model, num_bins=-1, save_path=None, return_full=False):
        if num_bins < 0:
            num_bins = self.num_class
        if save_path is not None:
            title = (save_path.split("-")[-1]).split(".")[0]
            bin_data, fig = _ece(model, self.testloader, num_bins, gen_fig=True, title=title)
            fig.savefig(save_path)
        else:
            bin_data = _ece(model, self.testloader, num_bins)
        
        for k, v in bin_data.items():
            if isinstance(v, np.ndarray):
                bin_data[k] = v.tolist()

        if return_full:
            return bin_data
        else:
            return bin_data["expected_calibration_error"]

    @deprecated(new="Metric.negative_flip_rate")
    def nfr(self, model1, model2):
        return self.negative_flip_rate(model1, model2)
    
    @deprecated(reason="It is not recommended to use this method as the name 'class_acc' is not standardized.", new="Metric.class_accuracy")
    def class_acc(self, model):
        return self.class_accuracy(model)

    @deprecated(reason="It is not recommended to use this method as the name 'acc' is not standardized.", new="Metric.accuracy")
    def acc(self, model):
        return self.accuracy(model)
