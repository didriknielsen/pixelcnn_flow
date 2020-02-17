class Callback(object):
    '''Callback base class'''

    def __init__(self):
        pass

    def on_train_begin(self, train_iterator, num_epochs):
        pass

    def on_epoch_begin(self, train_iterator, num_epochs, epoch):
        pass

    def on_batch_begin(self, train_iterator, num_epochs, epoch, iteration, batch):
        pass

    def on_batch_end(self, train_iterator, num_epochs, epoch, iteration, batch, loss):
        pass

    def on_epoch_end(self, train_iterator, num_epochs, epoch):
        pass

    def on_train_end(self, train_iterator, num_epochs):
        pass
