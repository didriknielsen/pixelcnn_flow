import os
import json
from prettytable import PrettyTable
import torch
from .callback import Callback


class MetricLoggingCallback(Callback):
    '''Callback for logging test metrics'''

    def __init__(self, test_fn, metric_names, test_every, log_folder=None):
        super(MetricLoggingCallback, self).__init__()
        self.test_fn = test_fn
        self.metric_names = metric_names
        self.test_every = test_every
        self.log_folder = log_folder
        self.result_dict = {metric_name: [] for metric_name in metric_names}
        self.epochs = []

    def on_epoch_end(self, epoch, **kwargs):
        if (epoch+1) % self.test_every == 0:
            self.epochs.append(epoch)
            eval_dict = self.test_fn()
            for metric_name, metric_value in eval_dict.items():
                self.result_dict[metric_name].append(metric_value)
            if self.log_folder: self.write_to_file()

    def write_to_file(self):
        with open(os.path.join(self.log_folder,'result_dict.json'), 'w') as f:
            json.dump(self.result_dict, f)
        with open(os.path.join(self.log_folder,'results.txt'), "w") as f:
            f.write(str(self.get_result_table()))

    @property
    def last_result_epoch(self):
        if len(self.epochs)>0:
            return self.epochs[-1]
        else:
            return -1

    def get_result_table(self):
        table = PrettyTable()
        table.add_column('Epoch', self.epochs)
        for metric_name, results in self.result_dict.items():
            table.add_column(metric_name, results)
        return table

    def get_last_result(self):
        return {metric_name: results[-1] for metric_name, results in self.result_dict.items()}
