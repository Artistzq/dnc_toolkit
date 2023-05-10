import numpy as np
import torch
from matplotlib import pyplot as plt
import sklearn.metrics as sm
from thop import profile
import functools
from . import reliability_diagrams as rd
from ..utils.decorators import deprecated


device = "cuda" if torch.cuda.is_available() else "cpu"

classes = {
    "CIFAR10":('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
}


def get_conv_linear_layers(net):
    names = []
    for name, layer in net.named_modules():
        if isinstance(layer, torch.nn.modules.conv.Conv2d) or isinstance(layer, torch.nn.modules.Linear):
            names.append(name)
    return names


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

@deprecated
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


def _nfr(model1, model2, data_loader):
    model1.eval()
    model2.eval()
    model1 = model1.to(device)
    model2 = model2.to(device)
    vals = []
    total = 0
    for X, y in data_loader:
        X = X.to(device)
        y = y.to(device)
        with torch.no_grad():
            pred1 = model1(X)
            pred2 = model2(X)
        pred_h1 = torch.argmax(pred1, axis=-1)
        pred_h2 = torch.argmax(pred2, axis=-1)
        vals.append(__nfr_compute(pred_h1, pred_h2, y).item())
        total += y.numel()
    return 100. * sum(vals) / total


def __nfr_compute(pred_h1: torch.Tensor, pred_h2: torch.Tensor, truth: torch.Tensor):
    correct_h1 = pred_h1.eq(truth)
    wrong_h2 = ~pred_h2.eq(truth)
    negative_flip = wrong_h2.masked_select(correct_h1)
    val = negative_flip.count_nonzero()
    return val                          
    # val = negative_flip.count_nonzero() / truth.size(0)
    # return 100. * val.item()


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


def _conf_matrix(model, data_loader):
    mat = None
    model.eval()
    model = model.to(device)
    for X, y in data_loader:
        with torch.no_grad():
            X = X.to(device)
            pred = model(X)
        y_pred = torch.argmax(pred, axis=-1).cpu().numpy()
        y_true = y.numpy()
        if mat is None:
            mat = sm.confusion_matrix(y_true, y_pred, labels=[i for i in range(100)])
        else:
            mat += sm.confusion_matrix(y_true, y_pred, labels=[i for i in range(100)])
    cmd = sm.ConfusionMatrixDisplay(mat)
    cmd.plot()
    print(mat)
    return mat, cmd


def _apfd(model, data_loader):
    model.eval()
    model = model.to(device)
    error_order_sum = 0
    total = 0
    error = 0
    for X, y in data_loader:
        with torch.no_grad():
            X = X.to(device)
            pred = model(X)
            y = y.to(device)
        diff = y - torch.argmax(pred, axis=-1)
        error += diff.count_nonzero().item()
        diff = diff.cpu().numpy()
        error_order = np.where(diff != 0)[0] + total + 1
        error_order_sum += np.sum(error_order)
        total += y.numel()
    return 1 - error_order_sum / (total * error) + 1 / (2 * total)


class Metric:
    def __init__(self, dataset, dataloader, num_class, use_gpu=True) -> None:
        self.testset, self.testloader, self.num_class = dataset, dataloader, num_class
        self.device = "cuda" if use_gpu else "cpu"
    
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
    
    @deprecated(reason="It is not recommended to use this method as the name 'acc' is not standardized.", new="Metric.accuracy")
    def acc(self, model):
        return self.accuracy(model)

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
                X = X.to(device)
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
    
    @deprecated(reason="It is not recommended to use this method as the name 'class_acc' is not standardized.", new="Metric.class_accuracy")
    def class_acc(self, model):
        return self.class_accuracy(model)

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

    def nfr(self, model1, model2):
        return _nfr(model1, model2, self.testloader)

    def clever(self, model, num_cases=100):
        
        from art.estimators.classification import PyTorchClassifier
        from art.metrics.metrics import clever_u
        
        torch.manual_seed(42)
        testset = torch.utils.data.random_split(self.testset, [num_cases, 10000-num_cases])[0]
        avg_score, scores = _robustness(model, testset, self.num_class)
        return avg_score
    
    def conf_matrix(self, model, save_path):
        mat, cmd = _conf_matrix(model, self.testloader)
        cmd.figure_.savefig(save_path)
        return mat
    
    def cka_matrix(self, model1, model2) -> np.ndarray:
        from torch_cka import CKA
        cka = CKA(model1, model2,
                model1_layers=get_conv_linear_layers(model1),
                model2_layers=get_conv_linear_layers(model2),
                device=device)

        cka.compare(self.testloader) # secondary dataloader is optional

        results = cka.export()  # returns a dict that contains model names, layer names
                                # and the CKA matrix
        results.pop("CKA")
        return cka.hsic_matrix, results
    
    def APFD(self, model, k, sort=True):
        # 从原始loader获取排序
        model.eval()
        model = model.to(device)
        probs = []
        for X, y in self.testloader:
            X = X.to(device)
            y = y.to(device)
            with torch.no_grad():
                pred = model(X)
            probs.append(pred)
        probs = torch.cat(probs).cpu()
        probs = torch.nn.functional.softmax(probs, dim=-1).detach().cpu().numpy()
        # 按照deepgini排序
        gini = np.sum(probs**2,axis=1)
        ranks = np.argsort(gini)[: k]
        if not sort:
            ranks = list(range(10000))[: k]
        # 构建新的loader
        class SortedSampler(torch.utils.data.Sampler):
            def __init__(self, ranks):
                self.ranks = ranks
        
            def __iter__(self):
                return iter(self.ranks)
        
            def __len__(self):
                return len(self.ranks)
        sampler = SortedSampler(ranks)
        new_loader = torch.utils.data.DataLoader(
            self.testset, 
            batch_size=100, 
            shuffle=False, 
            sampler=sampler,
            num_workers=4)
        if k < 0:
            new_loader = self.testloader
        apfd = _apfd(model, new_loader)
        acc = _acc(model, new_loader)
        ece = _ece(model, new_loader)["expected_calibration_error"]
        return round(apfd*100, 2)
        # return {"apfd": round(apfd*100, 2), "acc": round(100*acc, 2), "ece": round(ece*100, 2)}