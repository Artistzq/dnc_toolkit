import torch
from torch import nn
import torch.nn.functional as F
import copy


class PSKDLoss(nn.Module):

    def __init__(self, total_epoch, alpha_T, temperature=1, start_epoch=1):
        super(PS_KD, self).__init__()
        self.current_epoch = start_epoch
        self.total_epoch = total_epoch
        self.alpha_T = alpha_T
        self.alpha_t = alpha_T * self.current_epoch / self.total_epoch

        self.temperature = temperature

    def logist_inputs(self, inputs):
        self.inputs = inputs

    def forward(self, outputs, labels):
        """temperature = 1
        """
        # P_S = F.softmax(outputs / self.temperature, dim=-1) # (batch_size, num_class)
        if self.current_epoch == 1:
            kd_item = (1 - self.alpha_t + self.alpha_t * (self.temperature ** 2)) * \
                F.cross_entropy(outputs, labels)
        else:
            teacher_outputs = self.teacher(self.inputs).data
            # print(self.inputs.shape, outputs.shape)
            kd_item = (1 - self.alpha_t) * F.cross_entropy(outputs, labels) + \
                self.alpha_t * (self.temperature ** 2) * \
                    F.kl_div(
                        F.log_softmax(outputs / self.temperature, dim=1),
                        F.softmax(teacher_outputs / self.temperature, dim=1),
                        reduction="batchmean"
                    )
        return kd_item
    
    def logist_new_teacher(self, teacher, inputs):
        if self.current_epoch > 1:
            self.teacher = teacher
            self.inputs = inputs

    def update_params(self):
        print("Last epoch: {}, alpha: {}".format(self.current_epoch, self.alpha_t))
        self.current_epoch += 1
        print("Updated epoch: {}, alpha: {}".format(self.current_epoch, self.alpha_t))
        self.alpha_t = self.alpha_T * self.current_epoch / self.total_epoch
        self.alpha_t = max(0, self.alpha_t)