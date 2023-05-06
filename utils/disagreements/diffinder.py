import torch
import torch.nn as nn
from torchattacks.attack import Attack


class Diffinder(Attack):
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
    def __init__(self, model, ref_model, eps=8/255, alpha=2/255, steps=10, distance_restrict=8/255, random_start=True, normalization=None):
        
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ['default', 'targeted']
        self.ref_model = ref_model
        self.distance = distance_restrict
        # self.loss = nn.MSELoss()
        self.loss = nn.CrossEntropyLoss()
        if normalization is not None:
            self.set_normalization_used(normalization[0], normalization[1])

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
            outputs = self.get_logits(adv_images)
            
            ref_logits = self.get_logits(adv_images, self.ref_model)
            cost = self.loss(outputs, ref_logits) # MSE Loss
            
            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=images-self.distance, max=images+self.distance).detach()

        return adv_images

    def get_logits(self, inputs, model=None, labels=None, *args, **kwargs):
        if model is None:
            return super().get_logits(inputs, labels, *args, **kwargs)
        else:
            if self._normalization_applied is False:
                inputs = self.normalize(inputs)
            logits = model(inputs)
            return logits