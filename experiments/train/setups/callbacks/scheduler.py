from .callback import Callback


class SchedulerCallback(Callback):
    '''Scheduler callback'''

    def __init__(self, scheduler, every_iteration=False):
        super(SchedulerCallback, self).__init__()
        self.scheduler = scheduler
        self.every_iteration = every_iteration

    def on_epoch_end(self, **kwargs):
        if not self.every_iteration:
            self.scheduler.step()


    def on_batch_end(self, **kwargs):
        if self.every_iteration:
            self.scheduler.step()
