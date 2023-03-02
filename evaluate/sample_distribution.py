import numpy as np


class Method:
    def __init__(self, probs) -> None:
        self.probs = probs
        if not isinstance(probs, np.ndarray):
            self.probs = self.probs.cpu().numpy()
    

class DeepGini(Method):
    def cal(self):
        gini = 1 - np.sum(self.probs**2,axis=1)
        return gini


class Entropy(Method):
    def cal(self):
        return np.sum(self.probs * np.log(self.probs), axis=1)
    
    
class MaxDiffer(Method):
    def cal(self):
        maxs = np.max(self.probs, axis = 1)
        idx = np.argsort(self.probs, axis=1)
        # 获取倒数第二个索引
        second_largest_idx = idx[:, -2]
        # 次大值
        second_maxs = self.probs[range(len(self.probs)), second_largest_idx]
        return maxs - second_maxs
    

if __name__ == "__main__":
    res = DeepGini(np.array([
        [1, 2, 3],
        [3, 4, 1],
        [3, 1, 9],
        [5, 6,10]
    ])).cal()
    
    print(res)