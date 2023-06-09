import torch
import torch.nn as nn
from torchattacks.attack import Attack
import torch.nn.functional as F

from .finder import Finder
from .diffchaser import DiffChaser
from .utils import Uncertainty, RevLoss, wrapper


class Diffinder(Attack, Finder):
    r"""
    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack, white boxed.
        ref_model (nn.Module): a reference model with which model behave different, black boxed.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        distance_restrict (float): limit for finally perturbed images.
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, ref_model, ref_proxy=None, eps=8/255, alpha=2/255, steps=10, distance_restrict=8/255, random_start=True, normalization=None):
        
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ['default', 'targeted']
        self.ref_model = ref_model
        self.distance = distance_restrict
        # self.loss = nn.MSELoss()
        # self.loss = nn.CrossEntropyLoss()
        self.loss = nn.KLDivLoss(reduction = "batchmean")
        if normalization is not None:
            self.set_normalization_used(normalization[0], normalization[1])
        
        self.blacked = ref_model
        self.whited = model
        self.proxy = ref_proxy

    def forward(self, images, labels):
        r"""
        Overridden.

        Args:
            images (_type_): 接受的输入是：(N, C, H, W)或(C, H ,W)
            labels (_type_): 
            
        Returns:
            _type_: _description_
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        
        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            
            # 代理与black的loss尽可能小
            # whited与black的loss尽可能大

            # black_logits = self.get_logits(adv_images, self.ref_model)
            # proxy_logits = self.get_logits(adv_images, self.proxy)
            black_logits = self.get_logits(adv_images, self.ref_model)
            white_logits = self.get_logits(adv_images)

            # cost = nn.MSELoss()(black_logits, white_logits)
            cost = self.loss(
                F.log_softmax(white_logits/4 + 1e-8, dim=1), 
                F.softmax(black_logits/4, dim=1)
            )
            # cost2 = Uncertainty.margin(F.softmax(white_logits, dim=-1))
            # cost2 = - cost2.sum() / cost2.numel()

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, 0, 1).detach()

        return adv_images

    def get_logits(self, inputs, model=None, labels=None, *args, **kwargs) -> torch.Tensor:
        if model is None:
            return super().get_logits(inputs, labels, *args, **kwargs)
        else:
            if self._normalization_applied is False:
                inputs = self.normalize(inputs)
            logits = model(inputs)
            return logits
        
    def find(self, datasource, save_path=None):
        images, labels = wrapper.to_tensor(datasource)
        founded = self.__call__(images, labels)
        if save_path:
            torch.save(founded, save_path)
        return founded


class SameFinder(Diffinder):
    def __init__(self, model, ref_model, eps=8 / 255, alpha=2 / 255, steps=10, distance_restrict=8 / 255, random_start=True, normalization=None):
        super().__init__(model, ref_model, eps, alpha, steps, distance_restrict, random_start, normalization)
        self.loss = RevLoss(self.loss)