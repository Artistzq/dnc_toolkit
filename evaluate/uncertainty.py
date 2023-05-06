import torch
import numpy as np
import math

class Uncertainty:
    def __check(func):
        def wrapper(cls, probs, *args, **kwargs):
            cls.device = "cpu"
            
            # 转化为tensor计算
            if isinstance(probs, np.ndarray):
                probs = torch.Tensor(probs)

            assert probs[0].sum() < 1.01 and probs[0].sum() > 0.98, "The sum of probabilities for each sample should be equal to 1. Consider using softmax correctly."
            assert probs.ndim == 2, "Probs wrong shape: {}, shoud be (batch_size, class_num)".format(probs.shape)
            # 从计算图分离
            probs = probs.to(cls.device).detach()

            probs = func(cls, probs, **kwargs)
            return probs
        return wrapper
    
    @classmethod
    @__check
    def entropy(cls, probs, norm=False):
        entropy = - torch.sum(probs * torch.log(probs), axis=1)
        if norm:
            entropy /= math.log(probs.shape[1])
            entropy = torch.clamp(entropy, 0, 1)
        return entropy
        
    @classmethod
    @__check
    def gini(cls, probs):
        gini = 1 - torch.sum(probs**2,axis=1)
        return gini

    @classmethod
    @__check
    def margin(cls, probs):
        maxs = torch.max(probs, axis = 1)[0]
        idx = torch.argsort(probs, axis=1)
        # 获取倒数第二个索引
        second_largest_idx = idx[:, -2]
        # 次大值
        second_maxs = probs[range(len(probs)), second_largest_idx]
        return maxs - second_maxs
    
    @classmethod
    @__check
    def confidence(cls, probs):
        return torch.max(probs, axis=1)[0]
    