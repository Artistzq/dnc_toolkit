import torch
from ...pipeline import Archive


class Compressor:
    
    def __init__(self) -> None:
        self.archive = None
    
    def set_archive(self, archive: Archive):
        self.archive = archive
    
    def compress(self, model):
        model = self.process(model)
        if self.archive:
            torch.save(model, self.archive.get_weight_path())
        return model
    
    def process(self, model):
        raise NotImplementedError()