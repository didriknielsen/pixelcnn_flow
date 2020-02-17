from collections import OrderedDict

import math
import torch

from pixelflow.distributions import DataParallelDistribution

from .base_setup import BaseSetup
from .metrics import NegAvgLogLikelihood, BitsPerDim


class CategoricalImageFlowSetup(BaseSetup):

    def __init__(self, supervised_dataloader=True, data_bits=8):
        super(CategoricalImageFlowSetup, self).__init__()
        self.data_bits = data_bits
        self.supervised_dataloader = supervised_dataloader

    def model_to_device(self, args):
        self.model = self.model.to(self.device)
        self.dataparallel = False
        if len(args.gpus) > 1:
            print("Using data parallelism with devices:", args.gpus)
            self.model = DataParallelDistribution(self.model, device_ids=list(range(len(args.gpus))), dim=0)
            self.dataparallel = True

    def get_images(self, batch):
        if self.supervised_dataloader:
            images, _ = batch
        else:
            images = batch
        return images.to(self.device)

    def get_metric_dict(self):
        return dict(ll = NegAvgLogLikelihood(),
                    bpd = BitsPerDim(2**self.data_bits))

    @property
    def metric_names(self):
        return list(self.get_metric_dict().keys())

    def test_fn(self):
        # Set model in eval mode
        self.model.train(False)
        with torch.no_grad():
            metric_dict = self.get_metric_dict()
            for batch in self.test_loader:
                images = self.get_images(batch)
                log_probs = self.model(images) + math.log(2**self.data_bits) * images.shape[1:].numel() # For backwards compatability
                subpixels = images.shape[1:].numel()
                for metric in metric_dict.values():
                    metric.add_batch(log_probs, numel=subpixels)

            result_dict = OrderedDict()
            for metric_name, metric in metric_dict.items():
                result_dict[metric_name] = metric.get_value()
        return result_dict

    def objective_fn(self, batch):
        images = self.get_images(batch)
        log_probs = self.model(images) + math.log(2**self.data_bits) * images.shape[1:].numel() # For backwards compatability
        subpixels = images.shape[1:].numel()
        return - log_probs.mean() / subpixels
