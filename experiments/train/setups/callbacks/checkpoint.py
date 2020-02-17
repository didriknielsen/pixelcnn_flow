import os
import torch
from .callback import Callback


class CheckpointingCallback(Callback):
    '''Callback for storing checkpoints of model, optimizer and scheduler'''

    def __init__(self, checkpoint_folder, checkpoint_every, model=None, optimizer=None, scheduler=None):
        self.checkpoint_folder = checkpoint_folder
        self.checkpoint_every = checkpoint_every
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        super(CheckpointingCallback, self).__init__()

    def checkpoint(self):
        if self.model is not None: torch.save(self.model.state_dict(), os.path.join(self.checkpoint_folder,'model.pt'))
        if self.optimizer is not None: torch.save(self.optimizer.state_dict(), os.path.join(self.checkpoint_folder,'optimizer.pt'))
        if self.scheduler is not None: torch.save(self.scheduler.state_dict(), os.path.join(self.checkpoint_folder,'scheduler.pt'))

    def on_train_begin(self, **kwargs):
        if not os.path.exists(self.checkpoint_folder):
            os.makedirs(self.checkpoint_folder)

    def on_epoch_end(self, epoch, **kwargs):
        if (epoch+1) % self.checkpoint_every == 0:
            with open(os.path.join(self.checkpoint_folder,'checkpoint.txt'), 'w') as f:
                f.write("Epoch: "+str(epoch))
            self.checkpoint()

    def on_train_end(self, num_epochs, **kwargs):
        with open(os.path.join(self.checkpoint_folder,'checkpoint.txt'), 'w') as f:
            f.write("Training done. Final model after {} epochs stored.".format(num_epochs))
        self.checkpoint()
