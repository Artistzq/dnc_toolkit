import numpy as np
import torch
from matplotlib import pyplot as plt
import sklearn.metrics as sm
from thop import profile
from . import reliability_diagrams as rd

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


def _acc(model, data_loader):
    model.eval()
    model = model.to(device)
    error = 0
    total = 0
    for X, y in data_loader:
        with torch.no_grad():
            X = X.to(device)
            pred = model(X)
            y = y.to(device)
        diff = y - torch.argmax(pred, axis=-1)
        error += diff.count_nonzero().item()
        total += y.numel()
    return 1 - (error / total)


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

@DeprecationWarning
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


class Metric():
    def __init__(self, dataset, dataloader, num_class, use_gpu=True) -> None:
        self.testset, self.testloader, self.num_class = dataset, dataloader, num_class
        self.use_gpu = use_gpu
        self.device = "cuda" if use_gpu else "cpu"
    
    @classmethod
    def macs(cls, model, input_shape):
        """
        input_shape: (3, 32, 32)
        """
        input = torch.randn(1, input_shape[0], input_shape[1], input_shape[2])
        model.to("cpu")
        input.to("cpu")
        macs, params = profile(model, inputs=(input, ), verbose=False)
        macs = macs / 1048576 / 1024
        params = params / 1048576
        return round(macs, 2)
    
    @classmethod
    def flops(cls, model, input_shape):
        """
        input_shape: (3, 32, 32)
        """
        input = torch.randn(1, input_shape[0], input_shape[1], input_shape[2])
        model.to("cpu")
        input.to("cpu")
        macs, params = profile(model, inputs=(input, ), verbose=False)
        macs = macs / 1048576 / 1024
        params = params / 1048576
        return round(macs*2, 2), round(params, 2)
    
    def acc(self, model):
        return _acc(model, self.testloader)
    
    def class_acc(self, model):
        class_acc = _class_acc(model, self.testloader, self.num_class)
        return class_acc[:, 2].tolist()

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