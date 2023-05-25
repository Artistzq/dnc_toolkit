from .metrics import ModelMetric
from ..utils import Timer

formatter = "[{}]: {:.4f}"

def evaluate_kit(model, dataset, ref_model=None):
    metric = ModelMetric(dataset.test_loader, verbose=True)
    
    print(formatter.format("FLOPs", metric.flops(model, (3, 32, 32))))
    print(formatter.format("Params", metric.params(model, (3, 32, 32))))
    
    with Timer("Infer Time"):
        print(formatter.format("ACC", metric.accuracy(model)))
    
    print(formatter.format("ECE", metric.expect_calibration_error(model)))
    
    if ref_model:
        print(formatter.format("NFR", metric.negative_flip_rate(model, ref_model)))
        print(formatter.format("DR", metric.disagree_rate(model, ref_model)))
    