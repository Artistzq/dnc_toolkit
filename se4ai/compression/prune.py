import torch
import torch_pruning as tp

from .compressor import Compressor
from ...pipeline import TinyTrainer as Trainer
from ...evaluate import ModelMetric


class Pruner(Compressor):
    
    def __init__(self, dataset, sparsity, iterative_steps=5, retrainer: Trainer=None, device="cuda") -> None:
        self.dataset = dataset
        self.sparsity = sparsity
        self.iterative_steps = iterative_steps
        self.retrainer = retrainer
        self.device = device
    
    def process(self, model):
        metric = ModelMetric(self.dataset.test_loader)
        shape = metric.get_shape()
        
        example_inputs = torch.randn((1, ) + shape).to(self.device)
        imp = tp.importance.TaylorImportance()
        # DO NOT prune the final classifier!
        ignored_layers = []
        for module in model.modules():
            if isinstance(module, torch.nn.Linear) and module.out_features == self.dataset.num_classes:
                ignored_layers.append(module)
        # print(ignored_layers)
        
        # iterative_steps = 5 # progressive pruning
        pruner = tp.pruner.MagnitudePruner(
            model,
            example_inputs,
            importance=imp,
            iterative_steps=self.iterative_steps,
            ch_sparsity=self.sparsity, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
            ignored_layers=ignored_layers,
        )

        print("Before Pruned. FLOPs: {} G, Params: {} MB".format(ModelMetric.flops(model, shape, "GB"), ModelMetric.params(model, shape)))
        for i in range(self.iterative_steps):
            if isinstance(imp, tp.importance.TaylorImportance):
                # Taylor expansion requires gradients for importance estimation
                model = model.to(self.device)
                loss = model(example_inputs).sum() # a dummy loss for TaylorImportance
                loss.backward() # before pruner.step()
            pruner.step()
            # finetune your model here
            # finetune(model)
            if self.retrainer:
                model = self.retrainer.train(model, self.dataset.train_loader, self.dataset.test_loader, "SGD")
            print("Prune steps: [{}], FLOPs: {} G, Params: {} MB, Acc: {:.2f}%".format(
                i + 1, ModelMetric.flops(model, shape, "MB"), ModelMetric.params(model, shape), 100 * metric.accuracy(model)
            ))
        return model