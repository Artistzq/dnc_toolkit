from .cnn import CNN5Layer
from .vgg import vgg16_bn
from .vgg import vgg13_bn
from .vgg import vgg11_bn
from .vgg import vgg19_bn
from .densenet import densenet121
from .densenet import densenet161
from .densenet import densenet169
from .densenet import densenet201
from .googlenet import googlenet
from .inceptionv3 import inceptionv3
from .inceptionv4 import inceptionv4
from .inceptionv4 import inception_resnet_v2
from .xception import xception
from .cifar_style_resnet import resnet20
from .cifar_style_resnet import resnet32
from .cifar_style_resnet import resnet44
from .cifar_style_resnet import resnet56
from .cifar_style_resnet import resnet110
from .resnet import resnet18
from .resnet import resnet34
from .resnet import resnet50
from .resnet import resnet101
from .resnet import resnet152
from .preactresnet import preactresnet18
from .preactresnet import preactresnet34
from .preactresnet import preactresnet50
from .preactresnet import preactresnet101
from .preactresnet import preactresnet152
from .resnext import resnext50
from .resnext import resnext101
from .resnext import resnext152
from .shufflenet import shufflenet
from .shufflenetv2 import shufflenetv2
from .squeezenet import squeezenet
from .mobilenet import mobilenet
from .mobilenetv2 import mobilenetv2
from .nasnet import nasnet
from .attention import attention56
from .attention import attention92
from .senet import seresnet18
from .senet import seresnet34
from .senet import seresnet50
from .senet import seresnet101
from .senet import seresnet152
from .wideresidual import wideresnet
from .stochasticdepth import stochastic_depth_resnet18
from .stochasticdepth import stochastic_depth_resnet34
from .stochasticdepth import stochastic_depth_resnet50
from .stochasticdepth import stochastic_depth_resnet101


from . import models_with_adapter as ada

def get_ada_network(net, num_class, adapter=None, gpu=True):
    if net == "resnet20":
        net = ada.resnet20(num_class, adapter)
    elif net == "resnet110":
        net = ada.resnet110(num_class, adapter)
    elif 'WRN' in net:
        _, depth, widen_factor = net.split("-")
        depth = int(depth)
        widen_factor = int(widen_factor)
        net = ada.wideresnet(num_class, depth, widen_factor)
    elif net == "resnet18":
        net = ada.resnet18(num_class)
    
    if gpu == True:
        net = net.to("cuda")
    return net


def get_network(net, num_class, gpu=None):
    """ return given network
    """
    class Args:
        def __init__(self):
            self.net = net
            self.gpu = gpu
    args = Args()
    
    if args.net == 'cnn':
        net = CNN5Layer()
    elif args.net == 'vgg16':
        net = vgg16_bn()
    elif args.net == 'vgg13':
        net = vgg13_bn()
    elif args.net == 'vgg11':
        net = vgg11_bn()
    elif args.net == 'vgg19':
        net = vgg19_bn()
    elif args.net == 'densenet121':
        net = densenet121()
    elif args.net == 'densenet161':
        net = densenet161()
    elif args.net == 'densenet169':
        net = densenet169()
    elif args.net == 'densenet201':
        net = densenet201()
    elif args.net == 'googlenet':
        net = googlenet()
    elif args.net == 'inceptionv3':
        net = inceptionv3()
    elif args.net == 'inceptionv4':
        net = inceptionv4()
    elif args.net == 'inceptionresnetv2':
        net = inception_resnet_v2()
    elif args.net == 'xception':
        net = xception()
    elif args.net == 'resnet20':
        net = resnet20(num_class)
    elif args.net == 'resnet32':
        net = resnet32(num_class)
    elif args.net == 'resnet44':
        net = resnet44(num_class)
    elif args.net == 'resnet56':
        net = resnet56(num_class)
    elif args.net == 'resnet110':
        net = resnet110(num_class)
    elif args.net == 'resnet18':
        net = resnet18(num_class)
    elif args.net == 'resnet34':
        net = resnet34(num_class)
    elif args.net == 'resnet50':
        net = resnet50(num_class)
    elif args.net == 'resnet101':
        net = resnet101(num_class)
    elif args.net == 'resnet152':
        net = resnet152(num_class)
    elif args.net == 'preactresnet18':
        net = preactresnet18()
    elif args.net == 'preactresnet34':
        net = preactresnet34()
    elif args.net == 'preactresnet50':
        net = preactresnet50()
    elif args.net == 'preactresnet101':
        net = preactresnet101()
    elif args.net == 'preactresnet152':
        net = preactresnet152()
    elif args.net == 'resnext50':
        net = resnext50()
    elif args.net == 'resnext101':
        net = resnext101()
    elif args.net == 'resnext152':
        net = resnext152()
    elif args.net == 'shufflenet':
        net = shufflenet()
    elif args.net == 'shufflenetv2':
        net = shufflenetv2(num_class)
    elif args.net == 'squeezenet':
        net = squeezenet(num_class)
    elif args.net == 'mobilenet':
        net = mobilenet()
    elif args.net == 'mobilenetv2':
        net = mobilenetv2(num_class)
    elif args.net == 'nasnet':
        net = nasnet()
    elif args.net == 'attention56':
        net = attention56()
    elif args.net == 'attention92':
        net = attention92()
    elif args.net == 'seresnet18':
        net = seresnet18()
    elif args.net == 'seresnet34':
        net = seresnet34()
    elif args.net == 'seresnet50':
        net = seresnet50()
    elif args.net == 'seresnet101':
        net = seresnet101()
    elif args.net == 'seresnet152':
        net = seresnet152()
    elif 'WRN' in args.net:
        _, depth, widen_factor = args.net.split("-")
        depth = int(depth)
        widen_factor = int(widen_factor)
        net = wideresnet(num_class, depth, widen_factor)
    elif args.net == 'stochasticdepth18':
        net = stochastic_depth_resnet18()
    elif args.net == 'stochasticdepth34':
        net = stochastic_depth_resnet34()
    elif args.net == 'stochasticdepth50':
        net = stochastic_depth_resnet50()
    elif args.net == 'stochasticdepth101':
        net = stochastic_depth_resnet101()

    
    if args.gpu:
        net = net.cuda()
        
    return net
