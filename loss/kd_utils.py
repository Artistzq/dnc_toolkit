# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class KnowledgeDistillationLoss(nn.Module):
    
    def __init__(self, criterion, teacher=None, alpha=0.8, T=4):
        """知识蒸馏损失函数

        Args:
            criterion (_type_): nn.CrossEntropyCriterion
            teacher (nn.Module, optional): 教师模型. Defaults to None.
            alpha (float, optional): KL散度项的权重. Defaults to 0.8.
            T (int, optional): 交叉熵项的权重. Defaults to 4.
        """
        
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
    
    
class AdaptiveWeightedKnowledgeDistillationLoss(KnowledgeDistillationLoss):
    
    def __init__(self, criterion, teacher=None, alpha=0.8, T=4, method="teacher"):
        """知识蒸馏损失函数

        Args:
            criterion (_type_): nn.CrossEntropyCriterion
            teacher (nn.Module, optional): 教师模型. Defaults to None.
            alpha (float, optional): 该参数无用，由算法自适应计算。KL散度项的权重. Defaults to 0.8.
            T (int, optional): 交叉熵项的权重. Defaults to 4.
        """
        super().__init__(criterion, teacher, alpha, T)
        assert method == "teacher" or method == "student" or method == "both"
        self.method = method
    
    def __cal_alpha__(self, outputs, reverse=False):
        # 根据样本计算对应的alpha
        probs = torch.nn.functional.softmax(outputs, dim=-1).detach().cpu().numpy()
        entropy = -np.sum(probs * np.log(probs), axis=1) # 负号很重要
        if not reverse:
            alpha_batch = 1 - entropy / np.log(probs.shape[1])
        else:
            alpha_batch = entropy / np.log(probs.shape[1])
        alpha_batch = torch.from_numpy(alpha_batch).to(outputs.device)
        alpha_batch = torch.clip(alpha_batch, 0, 1)
        return alpha_batch
    
    def __call__(self, outputs, targets):
        if self.teacher is not None:
            
            teacher_outputs = self.teacher(self.inputs).detach()
            if self.method == "teacher":
                alpha_batch = self.__cal_alpha__(teacher_outputs)
            elif self.method == "student":
                alpha_batch = self.__cal_alpha__(outputs, reverse=True)
            else:
                alpha_batch = (self.__cal_alpha__(teacher_outputs) + self.__cal_alpha__(outputs, reverse=True)) / 2
            loss = self.kd_loss(outputs, targets, teacher_outputs, alpha_batch, self.T)
        else:
            loss = self.criterion(outputs, targets)
        return loss

    def kd_loss(self, outputs, labels, teacher_outputs, alpha_batch, T):
        KD_loss = 0
        for output, label, teacher_output, alpha in zip(outputs, labels, teacher_outputs, alpha_batch):
            KD_loss += (1. - alpha) * F.cross_entropy(output, label) + \
                nn.KLDivLoss(reduction="sum")(
                    F.log_softmax(output/T + 1e-8, dim=0), 
                    F.softmax(teacher_output/T, dim=0)
                ) * alpha * T * T
        return KD_loss / labels.shape[0]
