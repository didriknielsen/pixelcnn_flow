from tqdm import tqdm
from .callback import Callback


class ProgressBarCallback(Callback):
    '''Callback for printing progress bar'''

    def __init__(self, loss_logging_callback=None, metric_logging_callback=None):
        super(ProgressBarCallback, self).__init__()
        self.loss_logging_callback = loss_logging_callback
        self.metric_logging_callback = metric_logging_callback
        self.progbar = None

    def on_epoch_begin(self, train_iterator, num_epochs, epoch, **kwargs):
        self.progbar = tqdm(total=len(train_iterator),
                            unit='batch')
        self.progbar.set_description('Epoch [{:d}/{:d}]'.format(epoch+1, num_epochs))
        self.progbar.refresh()

    def on_epoch_end(self, epoch, **kwargs):
        print_dict = dict()
        if self.loss_logging_callback is not None:
            print_dict['loss'] = self.loss_logging_callback.get_last_result()
        if self.metric_logging_callback is not None:
            if self.metric_logging_callback.last_result_epoch == epoch:
                print_dict.update(self.metric_logging_callback.get_last_result())
            else:
                print_dict.update(dict.fromkeys(self.metric_logging_callback.result_dict.keys(), None))
        self.progbar.set_postfix(print_dict)
        self.progbar.refresh()
        self.progbar.close()

    def on_batch_begin(self, **kwargs):
        self.progbar.update(1)
        self.progbar.refresh()

    def on_batch_end(self, **kwargs):
        print_dict = dict()
        if self.loss_logging_callback is not None:
            print_dict['loss'] = self.loss_logging_callback.get_loss()
        if self.metric_logging_callback is not None:
            print_dict.update(dict.fromkeys(self.metric_logging_callback.result_dict.keys(), None))
        self.progbar.set_postfix(print_dict)
        self.progbar.refresh()
