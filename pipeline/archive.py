import os


class Archive:
    
    def __init__(self, root, data_name, model_name, save_interval=-1, **kwargs) -> None:
        self.save_interval = save_interval
        self.base_dir = os.path.join(root, data_name, model_name)
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        self.base_dir = os.path.join(self.base_dir, "[{}]-[{}]".format(data_name, model_name))
        self.key_and_values = kwargs

    def add_information(self, info):
        if info:
            self.base_dir += "-[Info={}]".format(info)
        return self
    
    def add_tag(self, **kwargs):
        for k, v in kwargs.items():
            self.key_and_values[k] = v
        return self

    def set_tag(self, **kwargs):
        return self.add_tag(**kwargs)

    def combine(self, kvs: dict):
        items = ["[{}={}]".format(k, v) for (k, v) in kvs.items()]
        return "-".join(items)

    def get_weight_path(self, **kwargs):
        path = self.base_dir
        if self.key_and_values:
            path += '-' + self.combine(self.key_and_values)
        if kwargs:
            path += "-" + self.combine(kwargs)
        return path + ".pth"

    def get_log_path(self, **kwargs):
        return self.get_weight_path(**kwargs)[:-4] + ".log"


if __name__ == "__main__":
    archive = Archive("./checkpoints", "Test", "resnet20")
    print(archive.get_weight_path(AAA=1))
    print(archive.get_log_path())
    archive.add_tag(TTT=11)
    print(archive.get_weight_path(AAA=1))
    print(archive.get_log_path())
    # archive.add_information("e200")
    # print(archive.get_weight_path())
    # print(archive.get_log_path())
    