from torch import nn


class RevLoss(nn.Module):
    def __init__(self, loss) -> None:
        super().__init__()
        self.loss = loss
    
    def forward(self, output, target):
        return - self.loss(output, target)

