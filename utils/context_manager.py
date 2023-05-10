import time

class Timer:
    def __init__(self, task=None, verbose=True, time_unit='s', decimal_places=3):
        """计算时长的上下文管理器
        
        Example:
            # 计算代码块执行的时间
            with Timer(time_unit="ms", verbose=True) as timer:
                time.sleep(1)
            # -----------------------------------
            # Output:
            # Time elapsed: 1001.041 ms
            # -----------------------------------
            print(time.get)
            # -----------------------------------
            # Output:
            # 1001.041
            # -----------------------------------

        Args:
            time_unit (str, optional): 时间单位. Defaults to 's'.
            decimal_places (int, optional): 保留小数. Defaults to 3.
            verbose (bool, optional): 是否输出. Defaults to False.
        """
        self.start_time = None
        self.elapsed_time = None
        if task is None:
            self.task = "Undefined"
        else:
            self.task = task
        self.unit = time_unit
        self.decimal_places = decimal_places
        self.verbose = verbose
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_time = time.time() - self.start_time
        if self.verbose:
            print(f"Task [{str(self.task)}] elapsed for [{self.get_elapsed_time()} {self.unit}]")
    
    def get_elapsed_time(self):
        if self.unit == 's':
            return round(self.elapsed_time, 3)
        elif self.unit == 'ms':
            return round(self.elapsed_time * 1000, 3)
        else:
            raise ValueError(f"Invalid unit '{self.unit}'")
