"""
The script demonstrates a simple example of using ART with PyTorch. The example train a small model on the MNIST dataset
and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train the model,
it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import numpy as np

from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
import torch


def generate(model, x, loss, input_shape, nb_classes, eps=0.3, eps_step=0.1, max_iter=100):
    # classifier = PyTorchClassifier(
    #     model=model,
    #     clip_values=(min_pixel_value, max_pixel_value),
    #     loss=criterion,
    #     optimizer=optimizer,
    #     input_shape=(1, 28, 28),
    #     nb_classes=10,
    # )
    classifier = PyTorchClassifier(model, loss, input_shape[1: 3], nb_classes)
    attack = ProjectedGradientDescent(estimator=classifier, eps=eps, eps_step=eps_step, max_iter=max_iter, verbose=False, batch_size=input_shape[0])
    x_test_adv = attack.generate(x.cpu().numpy())
    return torch.Tensor(x_test_adv)

