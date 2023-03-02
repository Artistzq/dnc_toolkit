import argparse
import json
import os

import numpy as np
import torch
import torchvision
from torchvision import transforms
import sklearn.metrics as sm

from .metrics import Metric

def compute(testset, testloader, num_class, net, test_metrics=None, origin_net=None):
    # Metrics
    all_metrics = ["acc", "ece", "ece_full", "ece_diagram", "nfr", "clever", "class_acc", "confusion_matrix"]
    if test_metrics is None:
        test_metrics = all_metrics
    if 'cka' in test_metrics or 'nfr' in test_metrics:
        if origin_net is None:
            raise Exception("Parameter 'origin_net' need to be specified for comparing.")
    metric = Metric(testset, testloader, num_class)

    # Evaluate
    # record = {"name": name}
    record = {}
    if "acc" in test_metrics:
        record["acc"] = round(metric.acc(net) * 100, 2)
    if "ece" in test_metrics:
        record["ece"] = round(metric.ece(net, num_bins=10) * 100, 2)
    if "nfr" in test_metrics:
        record["nfr"] = metric.nfr(net, origin_net)
    if "apfd" in test_metrics:
        record["apfd"] = apfd = metric.APFD(net, 10000, sort=True)
    if "ece_diagram" in test_metrics:
        model_type = (name.split("-")[-1]).split(".")[0]
        metric.ece(net, num_bins=10, save_path="{}/reliability-{}".format(model_dir, model_type))
    if "clever" in test_metrics:
        record["clever"] = metric.clever(net, num_cases=10)
    if "class_acc" in test_metrics:
        class_acc = metric.class_acc(net, 100)
        record["class_acc"] = class_acc
    if "ece_full" in test_metrics:
        record["ece_full"] = metric.ece(net, num_bins=10, return_full=True)
    if "confusion_matrix" in test_metrics:
        model_type = (name.split("-")[-1]).split(".")[0]
        save_path="{}/matrix-{}".format(model_dir, model_type)
        if mat is None:
            mat = metric.conf_matrix(net, save_path)
        else:
            mat = metric.conf_matrix(net, save_path)
        print(mat)
        # cmd = sm.ConfusionMatrixDisplay(mat)
        # cmd.plot()
        print("done")
        # sm.ConfusionMatrixDisplay(mat).plot().figure_.savefig(save_path)
    if "cka_matrix" in test_metrics:
        cka_matrix, cka_res = metric.cka_matrix(origin_net, net)
        np.save("{}/origin_{}.npy".format(model_dir, name.split("-")[-1][:-4]), cka_matrix)
        record["cka_res"] = cka_res
    # print(record)
    return record

    # res_save_path = "".join(weight_path.split("/")[:-1])
    # json.dump(records, open("{}/results.json".format(model_dir), "a"), indent=4)
