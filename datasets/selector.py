import numpy as np
from torch.utils.data import DataLoader, Dataset
from ..evaluate import Uncertainty
from ..evaluate import ModelMetric
from . import wrapper

class Selector:
    
    def select(self, datasource, k):
        raise NotImplementedError


class RandomSelector(Selector):
    
    def select(self, datasource, k):
        return np.random.permutation(len(datasource))[: k]


class UncertaintyBasedSelector(Selector):
    """使用方法
        indices = type(
            'EntropyBasedSelector', 
            (UncertaintyBasedSelector, ),  
            {"get_uncertainty": lambda self, probs: Uncertainty.entropy(probs)}
        )(model_ori).select(dataset, k, 4)
      
    """
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self.func = None
    
    def select(self, datasource, k, T=1):
        if isinstance(datasource, Dataset):
            datasource: DataLoader = wrapper.dataset_to_loader(datasource)
        
        metric = ModelMetric(datasource)
        metric.device = next(self.model.parameters())
        probs = metric.get_probs(self.model, temperature=T)
        
        values = self.get_uncertainty(probs)
        indices = np.argsort(values)
        return indices[: k]
        
    def get_uncertainty(self, probs):
        raise NotImplementedError
    
    def get_instance(func):
        """
        Args:
            func (_type_): 不确定性的计算方式，lambda self, probs: process(probs)

        Returns:
            UncertaintyBasedSelector: 基于不确定性的选择器
        """
        return type(
            'UnKnown',
            (UncertaintyBasedSelector, ),
            {"get_uncertainty": func}
        )
