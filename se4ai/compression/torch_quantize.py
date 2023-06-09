from .compressor import Compressor
import torch
import copy
from torch.ao.quantization import quantize_fx, get_default_qconfig
from torch import nn

class TorchQuantizer(Compressor):
    
    def __init__(self, calib_loader, verbose=True) -> None:
        super().__init__()
        self.calib_loader = calib_loader
        self.verbose = verbose
        
    def compress(self, model):
        model = self.process(model)
        if self.archive:
            torch.save(model.state_dict(), self.archive.get_weight_path())
        return model
    
    def process(self, model, ):
        calib_loader = self.calib_loader
        model_prepare = copy.deepcopy(model)
        model_prepare.eval()
        model_prepare = model_prepare.to("cpu")
        
        if self.verbose:
            print("Set quantization config...")
        # 设置量化方式
        qconfig_dict = {"": get_default_qconfig(backend='fbgemm')}
        model_prepare = quantize_fx.prepare_fx(model=model_prepare, qconfig_dict=qconfig_dict)
        model_prepare.eval()

        if self.verbose:
            print("Calibrating...")
        # 标定，确定Activation的量化范围
        if calib_loader is not None:
            for X, y in calib_loader:
                model_prepare(X)
        if self.verbose:
            print("Calibrated!")

        if self.verbose:
            print("Converting...")
        # 根据之前设置的量化方式以及标定计算的参数， 进行模型转换， FP32--> INT8
        quantized_model = quantize_fx.convert_fx(graph_module=model_prepare)
        quantized_model.eval()
        if self.verbose:
            print("Converted!")
        
        return quantized_model
    
    @classmethod
    def load_model(self, raw_model, weight_path):
        model_prepare = copy.deepcopy(raw_model)
        model_prepare.eval()
        model_prepare = model_prepare.to("cpu")
        
        # 设置量化方式
        qconfig_dict = {"": get_default_qconfig(backend='fbgemm')}
        model_prepare = quantize_fx.prepare_fx(model=model_prepare, qconfig_dict=qconfig_dict)
        model_prepare.eval()

        # 根据之前设置的量化方式以及标定计算的参数， 进行模型转换， FP32--> INT8
        quantized_model = quantize_fx.convert_fx(graph_module=model_prepare)
        quantized_model.eval()
        
        quantized_model.load_state_dict(torch.load(weight_path))
        
        quantized_model = quantized_model.cpu()
        return quantized_model


class QuantizedModel(nn.Module):
    def __init__(self, model_fp32):
        super(QuantizedModel, self).__init__()
        # QuantStub converts tensors from floating point to quantized.
        # This will only be used for inputs.
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point.
        # This will only be used for outputs.
        self.dequant = torch.quantization.DeQuantStub()
        # FP32 model
        self.model_fp32 = copy.deepcopy(model_fp32)
        
        self.fuse_model()
        
    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.model_fp32(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        return x

    def set_qconfig(self, qconfig):
        self.qconfid = qconfig

    def fuse_model(self):
        """融合所有的conv层和bn层
        """
        # 找出相连的Conv层和BN层+Relu层
        named_modules = self.flattened_named_modules(self.model_fp32)
        processed_moduels_names = []
        fused_moduel_names = []
        for i in range(len(named_modules) - 2):
            window = named_modules[i: i + 3]
            names = [item[0] for item in window]
            if isinstance(window[0][1], nn.Conv2d) and isinstance(window[1][1], nn.BatchNorm2d) and isinstance(window[2][1], nn.ReLU):
                fused_moduel_names.append(names)
                processed_moduels_names.extend(names)
                i += 2

        named_modules = [item for item in named_modules if item[0] not in processed_moduels_names]
        for i in range(len(named_modules) - 1):
            window = named_modules[i: i + 2]
            names = [item[0] for item in window]
            if isinstance(window[0][1], nn.Conv2d) and isinstance(window[1][1], nn.BatchNorm2d):
                fused_moduel_names.append(names)
                i += 1
        
        torch.quantization.fuse_modules(self.model_fp32, fused_moduel_names, inplace=True)

    def flattened_named_modules(self, model: torch.nn.Module):
        flattened = []
        for name, module in model.named_modules():
            if list(module.children()) == []:
                flattened.append((name, module))
        return flattened
    
    def calibrate(self, loader, device=torch.device("cpu:0")):
        model = self.model_fp32
        model.to(device)
        model.eval()

        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            _ = self.forward(inputs)

