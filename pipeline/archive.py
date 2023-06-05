import os


class Archive:
    
    def __init__(self, root, data_name, model_name, save_interval=-1, **kwargs) -> None:
        self.save_interval = save_interval
        self.base_dir = os.path.join(root, data_name, model_name)
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        self.base_dir = os.path.join(self.base_dir, "[{}]-[{}]".format(data_name, model_name))
        self.base_dir += self.__combine(**kwargs)
    
    def __combine(self, **kwargs):
        add = ""
        for k, v in kwargs.items():
            add += "-[{}={}]".format(k, v)
        return add
    
    def add_information(self, info):
        if info:
            self.base_dir += "-[Info={}]".format(info)
        return self
    
    def add_tag(self, **kwargs):
        self.base_dir += self.__combine(**kwargs)
        return self
    
    def get_weight_path(self, **kwargs):
        return self.base_dir + self.__combine(**kwargs) + ".pth"
    
    def get_log_path(self, **kwargs):
        return self.base_dir + self.__combine(**kwargs) + ".log"
    
if __name__ == "__main__":
    archive = Archive("./checkpoints", "CIFAR10", "resnet20", save_interval=200, a=1, T=3)
    print(archive.get_weight_path())
    print(archive.get_log_path())
    archive.add_information("e200")
    print(archive.get_weight_path())
    print(archive.get_log_path())
    