from torchattacks import CW
import torch
import torch.nn as nn
from torch import optim

from .finder import Finder
from .utils import wrapper


class CWDiffinder(CW, Finder):
    
    def __init__(self, white_model, black_model, c=1, kappa=0, steps=50, lr=0.01):
        super(CWDiffinder, self).__init__(white_model, c, kappa, steps, lr)
        self.black_model = black_model
    
    def find(self, datasource, save_path=None):
        images, labels = wrapper.to_tensor(datasource)
        founded = self.__call__(images, labels)
        if save_path:
            torch.save(founded, save_path)
        return founded
    
    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        # w = torch.zeros_like(images).detach() # Requires 2x times
        w = self.inverse_tanh_space(images).detach()
        w.requires_grad = True

        best_adv_images = images.clone().detach()
        best_L2 = 1e10*torch.ones((len(images))).to(self.device)
        prev_cost = 1e10
        dim = len(images.shape)

        MSELoss = nn.MSELoss(reduction='none')
        Flatten = nn.Flatten()

        optimizer = optim.Adam([w], lr=self.lr)

        for step in range(self.steps):
            # Get adversarial images
            adv_images = self.tanh_space(w)

            # Calculate loss
            current_L2 = MSELoss(Flatten(adv_images),
                                 Flatten(images)).sum(dim=1)
            L2_loss = current_L2.sum()

            outputs = self.get_logits(adv_images)
            black_labels = self.get_logits(adv_images, self.black_model).argmax(dim=-1)
            
            if self.targeted:
                f_loss = self.f(outputs, target_labels).sum()
            else:
                f_loss = self.f(outputs, black_labels).sum()

            cost = L2_loss + self.c*f_loss

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # Update adversarial images
            _, pre = torch.max(outputs.detach(), 1)
            correct = (pre == black_labels).float()

            # filter out images that get either correct predictions or non-decreasing loss, 
            # i.e., only images that are both misclassified and loss-decreasing are left 
            mask = (1-correct)*(best_L2 > current_L2.detach())
            best_L2 = mask*current_L2.detach() + (1-mask)*best_L2

            mask = mask.view([-1]+[1]*(dim-1))
            best_adv_images = mask*adv_images.detach() + (1-mask)*best_adv_images

            # Early stop when loss does not converge.
            # max(.,1) To prevent MODULO BY ZERO error in the next step.
            if step % max(self.steps//10,1) == 0:
                if cost.item() > prev_cost:
                    return best_adv_images
                prev_cost = cost.item()

        return best_adv_images
    
    def get_logits(self, inputs, from_model=None, labels=None, *args, **kwargs) -> torch.Tensor:
        if from_model is None:
            return super().get_logits(inputs, labels, *args, **kwargs)
        else:
            if self._normalization_applied is False:
                inputs = self.normalize(inputs)
            logits = from_model(inputs)
            return logits