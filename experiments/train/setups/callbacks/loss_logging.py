import os
import math
from .callback import Callback


class LossLoggingCallback(Callback):
    '''Callback for logging training loss'''

    def __init__(self, log_folder=None):
        super(LossLoggingCallback, self).__init__()
        self.log_folder = log_folder
        if self.log_folder: self.create_loss_file()
        self.losses = []
        self.inf_counts = []
        self.nan_counts = []

    def on_epoch_begin(self, **kwargs):
        self.loss = 0.0
        self.batch_count = 0
        self.inf_count = 0
        self.nan_count = 0

    def on_epoch_end(self, epoch, **kwargs):
        self.losses.append(self.get_loss())
        self.inf_counts.append(self.inf_count)
        self.nan_counts.append(self.nan_count)
        if self.log_folder:self.append_loss_file(epoch)

    def on_batch_end(self, loss, **kwargs):
        batch_loss = loss.detach().cpu().item()
        if math.isinf(batch_loss):
            self.inf_count += 1
        elif math.isnan(batch_loss):
            self.nan_count += 1
        else:
            self.loss += batch_loss
            self.batch_count += 1

    def create_loss_file(self):
        with open(os.path.join(self.log_folder,'loss.txt'), 'w') as f:
            f.write("Epoch\tLoss\tInfs\tNans\n")

    def append_loss_file(self, epoch):
        with open(os.path.join(self.log_folder,'loss.txt'), 'a') as f:
            f.write("{}\t{}\t{}\t{}\n".format(epoch, self.losses[-1], self.inf_counts[-1], self.nan_counts[-1]))

    def get_loss(self):
        if self.batch_count > 0:
            return self.loss / self.batch_count
        else:
            return float('inf')

    def get_last_result(self):
        return self.losses[-1]
