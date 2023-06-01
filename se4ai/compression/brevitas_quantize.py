import torch
from brevitas.graph.calibrate import calibration_mode, bias_correction_mode
from brevitas.graph.quantize import preprocess_for_quantize
from brevitas.graph.target.flexml import preprocess_for_flexml_quantize
from brevitas_examples.imagenet_classification.ptq.ptq_common import quantize_model

from .compressor import Compressor
from ...datasets import wrapper
from ...datasets.cv_datasets import Dataset


class BrevitasQuantArgs:
    def __init__(self) -> None:
        self.target_backend = "generic"
        self.graph_eq_iterations = 20
        self.graph_eq_merge_bias = True
        self.act_bit_width = 8
        self.weight_bit_width = 8
        self.weight_narrow_range = True
        self.bias_bit_width = 'int32'
        self.scaling_per_output_channel = True
        self.act_quant_percentile = 99.999
        self.act_quant_type = 'symmetric'
        self.scale_factor_type = 'float32'


class BrevitasQuantizer(Compressor):
    
    def __init__(self, dataset: Dataset, calib=True, calibrate_percent=0.1, args=None) -> None:
        train_data = wrapper.loader_to_tensor(dataset.train_loader)
        calib_data = train_data[0][: int(calibrate_percent * dataset.num_train)]
        calib_labels = train_data[1][: int(calibrate_percent * dataset.num_train)]
        self.calib_loader = wrapper.tensor_to_loader(calib_data, calib_labels)
        
        self.args = args
        self.calib = calib
        if not args:
            self.args = BrevitasQuantArgs()
        
    def compress(self, model):
        args = self.args
        
        if args.target_backend == 'flexml':
            # flexml requires static shapes, pass a representative input in
            img_shape = 32
            processed_model = preprocess_for_flexml_quantize(
                model,
                torch.ones(1, 3, img_shape, img_shape),
                equalize_iters=args.graph_eq_iterations,
                equalize_merge_bias=args.graph_eq_merge_bias)
        elif args.target_backend == 'generic' or args.target_backend == 'layerwise':
            processed_model = preprocess_for_quantize(
                model,
                equalize_iters=args.graph_eq_iterations,
                equalize_merge_bias=args.graph_eq_merge_bias)
        else:
            raise RuntimeError(f"{args.target_backend} backend not supported.")

        print("==> Quantizing...")
        quant_model = quantize_model(
            processed_model,
            backend=args.target_backend,
            act_bit_width=args.act_bit_width,
            weight_bit_width=args.weight_bit_width,
            weight_narrow_range=args.weight_narrow_range,
            bias_bit_width=args.bias_bit_width,
            scaling_per_output_channel=args.scaling_per_output_channel,
            act_quant_percentile=args.act_quant_percentile,
            act_quant_type=args.act_quant_type,
            scale_factor_type=args.scale_factor_type)
        
        print("==> Quantized!")
        if self.calib:
            print("==> Calibrating...")
            self.calibrate(quant_model, self.calib_loader)
            print("==> Calibrated!")
        
        return quant_model


    def calibrate(self, model, calib_loader, bias_corr=True):
        """
        Perform calibration and bias correction, if enabled
        """
        model.eval()
        dtype = next(model.parameters()).dtype
        device = next(model.parameters()).device
        with torch.no_grad():
            with calibration_mode(model):
                for i, (images, target) in enumerate(calib_loader):
                    images = images.to(device)
                    images = images.to(dtype)
                    model(images)

            if bias_corr:
                with bias_correction_mode(model):
                    for i, (images, target) in enumerate(calib_loader):
                        images = images.to(device)
                        images = images.to(dtype)
                        model(images)
