import os
import argparse
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
from toolkit.datasets import wrapper as data_wrapper
from toolkit.datasets.selector import Selector, UncertaintyBasedSelector

from torch.utils.tensorboard import SummaryWriter

from conf import settings
from toolkit.utils.progress_bar import progress_bar


class Trainer:
    
    def __init__(self, 
        net, 
        criterion, 
        device,  
        batch_size,
        optimizer, 
        test_criterion=None,
        tensorboard=None, 
        ):
        
        self.net = net
        self.device = device
        self.loss_function = criterion
        self.tensorboard = tensorboard
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.test_criterion = test_criterion if test_criterion is not None else nn.CrossEntropyLoss()
        if tensorboard:
            self.__init_tensorboard()

    
    def __init_tensorboard(self):
        self.writer = SummaryWriter(log_dir=os.path.join(settings.LOG_DIR, args.net, settings.TIME_NOW + "_" + args.add_info))
        pass
        
    def __loss_function_ops(self, images, labels):
        pass
    
    def fit(self, model, training_loader, test_loader, epoch_nums):
        pass
    
    def __train_single_epoch(self, epoch, training_loader, verbose=True):
        net = self.net
        optimizer = self.optimizer
        writer = self.writer
        
        start = time.time()
        print('======> Training Network.....')
        net.train()
        
        correct = 0
        total = 0
        train_loss = 0
        
        for batch_index, (images, labels) in enumerate(training_loader):
            
            labels = labels.to(self.device)
            images = images.to(self.device)

            # 预留的接口
            self.__loss_function_ops(images, labels)

            optimizer.zero_grad()
            outputs = net(images)
            
            # with torch.autograd.detect_anomaly():
            loss = self.loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # tensorboard
            if self.tensorboard:
                n_iter = (epoch - 1) * len(training_loader) + batch_index + 1
                last_layer = list(net.children())[-1]
                for name, para in last_layer.named_parameters():
                    if 'weight' in name:
                        writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
                    if 'bias' in name:
                        writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)
            
            # 统计数据
            train_loss += loss.item()
            total += labels.size(0)
            correct += outputs.max(1)[1].eq(labels).sum().item()
            msg = 'Epoch: {:3} | Loss: {:0.4f} | ACC: {:0.2f} | LR: {:0.6f} | [{trained_samples}/{total_samples}]'.format(
                epoch,
                loss.item(),
                100.0 * correct / total,
                optimizer.param_groups[0]['lr'],
                trained_samples=batch_index * self.batch_size + len(images),
                total_samples=len(training_loader.dataset)
            )
            progress_bar(batch_index, len(training_loader), msg)

            # tensorboard
            if self.tensorboard:
                writer.add_scalar('Train/loss', loss.item(), n_iter)

            if epoch <= args.warm:
                warmup_scheduler.step()

        finish = time.time()

        # tensorboard
        if self.tensorboard:
            writer.add_scalar('Train/Accuracy', correct /  total, epoch)
            writer.add_scalar('Train/LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
        return 100.0 * correct / total, train_loss / len(training_loader.dataset)

    @torch.no_grad()
    def __test_single_epoch(self, epoch, test_loader):
        self.net.eval()
        print('======> Evaluating Network.....')
        test_loss = 0.0
        correct = 0.0
        total = 0

        for idx, (images, labels) in enumerate(test_loader):

            labels = labels.to(self.device)
            images = images.to(self.device)

            outputs = self.net(images)
            loss = self.test_criterion(outputs, labels)

            test_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
            msg = 'Epoch: {:3} | Loss: {:.4f} | ACC: {:.2f}'.format(
                epoch,
                test_loss / len(test_loader.dataset),
                100.0 * correct / total,
            )
            progress_bar(idx, len(test_loader), msg)
        print()
        
        if self.tensorboard:
            writer.add_scalar('Test/Average loss', test_loss / len(test_loader.dataset), epoch)
            writer.add_scalar('Test/Accuracy', correct / len(test_loader.dataset), epoch)

        return 100.0 * correct / len(test_loader.dataset), test_loss / len(test_loader.dataset)

    def select_data(self, datasource, k, selector: Selector, as_loader=False):
        if isinstance(datasource, DataLoader):
            datasource = data_wapper.loader_to_dataset(datasource)
        # datasource现在是个Dataset
        indices = selector.select(datasource, k)
        selected = torch.utils.data.Subset(datasource, indices)
        if as_loader:
            return data_wapper.dataset_to_loader(selected, self.batch_size, shuffle=False)
        return selected


if __name__ == '__main__':
    pass