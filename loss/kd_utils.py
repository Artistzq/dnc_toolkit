# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class KnowledgeDistillationLoss(nn.Module):
    
    def __init__(self, criterion, teacher=None, alpha=0.8, T=4):
        super(KnowledgeDistillationLoss, self).__init__()
        self.teacher = teacher
        self.criterion = criterion
        self.alpha = alpha
        self.T = T
        
    def logist_inputs(self, inputs):
        self.inputs = inputs
 
    def __call__(self, outputs, targets):
        # print(outputs)
        if self.teacher is not None:
            teacher_outputs = self.teacher(self.inputs).detach()
            loss = self.kd_loss(outputs, targets, teacher_outputs, self.alpha, self.T)
        else:
            loss = self.criterion(outputs, targets)
        # print(loss)
        return loss

    def kd_loss(self, outputs, labels, teacher_outputs, alpha, T):
        # print(outputs.shape)
        # print(teacher_outputs.shape)
        KD_loss = (1. - alpha) * F.cross_entropy(outputs, labels) + \
            nn.KLDivLoss(reduction = "batchmean")(
                F.log_softmax(outputs/T + 1e-8, dim=1), 
                F.softmax(teacher_outputs/T, dim=1)
            ) * alpha * T * T
        return KD_loss
