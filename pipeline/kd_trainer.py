from toolkit.pipeline.archive import Archive
from .tiny_trainer import TinyTrainer
from ..loss import KnowledgeDistillationLoss

class KDTrainer(TinyTrainer):
    def __init__(self, teacher, alpha, temperature, lr, num_epochs, weight_decay=0.0005, criterion=None, lr_milestone="default", device="cuda", archive: Archive = None, save_best=True) -> None:
        super().__init__(lr, num_epochs, weight_decay, criterion, lr_milestone, device, archive, save_best)
        self.criterion = KnowledgeDistillationLoss(self.criterion, teacher, alpha, temperature)
    
    def before_loss(self, loss, inputs, labels):
        self.criterion.logist_inputs(inputs)