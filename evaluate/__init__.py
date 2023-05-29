from .metrics import Metric, ModelMetric
from .uncertainty import Uncertainty
from .utils import Timer


formatter = "[{}]: {:.4f} {}"

def evaluate_kit(model, dataloader, ref_model=None):
    
    device = next(model.parameters()).device
    if device.type == "cuda":
        use_gpu = True
    else:
        use_gpu = False
    
    metric = ModelMetric(dataloader, use_gpu, verbose=True)
    shape = metric.get_shape()
    
    print(formatter.format("FLOPs", metric.flops(model, shape, "GB"), "GB"))
    print(formatter.format("Params", metric.params(model, shape), "MB"))
    
    with Timer("Infer Time"):
        print(formatter.format("ACC", metric.accuracy(model)*100, "%"))
    
    print(formatter.format("ECE", metric.expect_calibration_error(model) * 100, "%"))
    
    if ref_model:
        print(formatter.format("NFR", metric.negative_flip_rate(model, ref_model)*100, "%"))
        print(formatter.format("DR", metric.disagree_rate(model, ref_model)*100, "%"))
    