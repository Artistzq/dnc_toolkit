from typing import Any, Mapping
import torch.nn as nn
import torch.nn.functional as F


class Adapter(nn.Module):
    def __init__(self, end_features, mid_features=None):
        super(Adapter, self).__init__()
        if mid_features is None:
            mid_features = end_features // 2
        self.down = nn.Linear(end_features, mid_features)
        self.relu = nn.ReLU()
        # self.gelu = nn.GELU()
        self.up = nn.Linear(mid_features, end_features)
        nn.init.normal_(self.down.weight, 0, 0.01)
        nn.init.normal_(self.up.weight, 0, 0.01)

    def forward(self, x):
        out = self.down(x)
        out = self.relu(out)
        # out = self.gelu(out)
        out = self.up(out)
        return x + out


class AdapterFreezer(nn.Module):
    
    def freeze_weights(self):
        for name, parameter in self.named_parameters():
            if "adapter" not in name:
                parameter.requires_grad = False
    
    def unfreeze_weights(self):
        for name, parameter in self.named_parameters():
            if "adapter" not in name:
                parameter.requires_grad = True



from . import resnet
from . import cifar_style_resnet as csr

class C_ResNet(csr.ResNet, AdapterFreezer):
    def __init__(self, block, num_blocks, num_classes=10, adapter=None):
        super(ResNet, self).__init__(block, num_blocks, num_classes)
        self.adapter = adapter
        self.freeze_weights()
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        # out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        if self.adapter:
            out = self.adapter(out)
        out = self.linear(out)
        return out
    

def resnet20(num_classes, adapter=None):
    return C_ResNet(csr.BasicBlock, [3, 3, 3], num_classes, adapter)


def resnet32(num_classes, adapter=None):
    return C_ResNet(csr.BasicBlock, [5, 5, 5], num_classes=num_classes)


def resnet44(num_classes):
    return C_ResNet(csr.BasicBlock, [7, 7, 7], num_classes=num_classes)


def resnet56(num_classes):
    return C_ResNet(csr.BasicBlock, [9, 9, 9], num_classes=num_classes)

def resnet110(num_classes, adapter=None):
    return C_ResNet(csr.BasicBlock, [18, 18, 18], num_classes=num_classes, adapter=adapter)

def resnet1202(num_classes):
    return C_ResNet(csr.BasicBlock, [200, 200, 200], num_classes=num_classes)


class ResNet(resnet.ResNet, AdapterFreezer):
    def __init__(self, block, num_blocks, num_classes=10, adapter=None):
        super(ResNet, self).__init__(block, num_blocks, num_classes)
        self.adapter = adapter
        if self.adapter is None:
            self.adapter = Adapter(self.fc.in_features)
        self.freeze_weights()
    
    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.adapter(output)
        output = self.fc(output)

        return output
    
def resnet18(num_classes=200):
    """ return a ResNet 18 object
    """
    return ResNet(resnet.BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

def resnet34(num_classes=200):
    """ return a ResNet 34 object
    """
    return ResNet(resnet.BasicBlock, [3, 4, 6, 3], num_classes=num_classes)

def resnet50(num_classes=200):
    """ return a ResNet 50 object
    """
    return ResNet(resnet.BottleNeck, [3, 4, 6, 3], num_classes=num_classes)

def resnet101(num_classes=200):
    """ return a ResNet 101 object
    """
    return ResNet(resnet.BottleNeck, [3, 4, 23, 3], num_classes=num_classes)

def resnet152(num_classes=200):
    """ return a ResNet 152 object
    """
    return ResNet(resnet.BottleNeck, [3, 8, 36, 3], num_classes=num_classes)


from . import wideresidual as wrn

class WideResNet(wrn.WideResNet, AdapterFreezer):
    def __init__(self, num_classes, block, depth=50, widen_factor=1, adapter=None):
        super().__init__(num_classes, block, depth, widen_factor)
        self.adapter = adapter
        if self.adapter is None:
            self.adapter = Adapter(self.linear.in_features)
        self.freeze_weights()
    
    def forward(self, x):
        x = self.init_conv(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.adapter(x)
        x = self.linear(x)
        return x

def wideresnet(num_classes=100, depth=40, widen_factor=10):
    net = WideResNet(num_classes, wrn.WideBasic, depth=depth, widen_factor=widen_factor)
    return net


