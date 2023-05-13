import logging


class Logger:
    def __init__(self, name, logfile):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # 输出到控制台的handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # 输出到文件的handler
        fh = logging.FileHandler(logfile, mode='a')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        self.info("Logging: {}".format(name))

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)
