import os
import sys
import time
import pickle
import argparse
import importlib
from collections import OrderedDict
from prettytable import PrettyTable

import torch
import torch.nn as nn

from .utils import set_seeds, set_env
from .callbacks import LossLoggingCallback, MetricLoggingCallback, ProgressBarCallback, SchedulerCallback, CheckpointingCallback

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(importlib.util.find_spec("pixelflow").origin)), 'experiments') # [pixelcnn_flow/experiments/.]


class BaseSetup(object):

    def __init__(self):
        self.env_initialized = False

    def _get_default_folder(self, folder_name):
        script_path = os.path.realpath(sys.argv[0])
        p = script_path.split(DEFAULT_PATH)
        assert len(p) == 2, 'Script need to be in {}'.format(DEFAULT_PATH)
        p = p[1].split('.py')
        script_id = p[0][1:]
        folder = os.path.join(DEFAULT_PATH, folder_name, script_id)
        return os.path.join(folder, time.strftime("%Y-%m-%d_%H-%M-%S"))

    def get_parser(self):

        parser = argparse.ArgumentParser()

        # Path params
        parser.add_argument('--log_folder', type=str, default=self._get_default_folder('log'))
        parser.add_argument('--check_folder', type=str, default=self._get_default_folder('check'))

        # General params
        parser.add_argument('--gpus', type=eval, default=[])
        parser.add_argument('--seed', type=int, default=123)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--pin_memory', type=eval, default=False)

        # Training params
        parser.add_argument('--num_epochs', type=int, default=10)
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--grad_clip_norm', type=float, default=None)
        parser.add_argument('--grad_clip_value', type=float, default=None)

        # Eval params
        parser.add_argument('--test_every', type=int, default=None)

        # Check params
        parser.add_argument('--checkpoint_every', type=int, default=None)
        parser.add_argument('--checkpoint_model_only', type=eval, default=False)

        return parser

    def prepare_env(self, args):

        print("Setting environment variables...")
        set_env(args.gpus, verbose=True)

        print("Setting seeds...")
        set_seeds(args.seed)

        print("Defining devices...")
        self.define_devices(args)

        self.env_initialized = True

    def register_data(self, train_set, test_set):
        self.train_set = train_set
        self.test_set = test_set

    def register_model(self, model):
        self.model = model

    def register_optimizer(self, optimizer, scheduler=None, decay_every_iteration=False):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.decay_every_iteration = decay_every_iteration

    def define_devices(self, args):
        self.device = 'cuda' if torch.cuda.is_available() and len(args.gpus) > 0 else 'cpu'
        print("Using device:", self.device)

    def model_to_device(self, args):
        self.model = self.model.to(self.device)
        self.dataparallel = False
        if len(args.gpus) > 1:
            print("Using data parallelism with devices:", args.gpus)
            self.model = nn.DataParallel(self.model, device_ids=list(range(len(args.gpus))), dim=0)
            self.dataparallel = True

    def prepare_data_loaders(self, args):
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=args.batch_size, shuffle=True, pin_memory=args.pin_memory, num_workers=args.num_workers)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=args.batch_size, shuffle=False, pin_memory=args.pin_memory, num_workers=args.num_workers)

    def store_arguments(self, args):

        # Create log folder
        os.makedirs(args.log_folder)
        print("Storing logs in:", args.log_folder)

        if args.checkpoint_every:
            os.makedirs(args.check_folder)
            print("Storing checkpoints in:", args.check_folder)

        # Store parameters
        args_dict = vars(args)
        args_table = PrettyTable(['Arg', 'Value'])
        for arg, val in args_dict.items():
            args_table.add_row([arg, val])
        with open(os.path.join(args.log_folder,'args.pickle'), "wb") as f:
            pickle.dump(args, f)
        with open(os.path.join(args.log_folder,'args.txt'), "w") as f:
            f.write(str(args_table))

    @property
    def metric_names(self):
        raise NotImplementedError()

    def test_fn(self):
        raise NotImplementedError()

    def objective_fn(self, batch):
        raise NotImplementedError()

    def define_training(self, args):

        self.callbacks = OrderedDict()

        log_folder = args.log_folder

        # Callback for logging training loss
        self.callbacks['loss'] = LossLoggingCallback(log_folder=log_folder)
        if args.test_every:
            # Callback for logging test metrics
            self.callbacks['metric'] = MetricLoggingCallback(test_fn=self.test_fn, metric_names=self.metric_names, test_every=args.test_every, log_folder=log_folder)
            metric_logging_callback = self.callbacks['metric']
        else:
            metric_logging_callback = None
        # Callback for printing progress bar
        self.callbacks['progbar'] = ProgressBarCallback(loss_logging_callback=self.callbacks['loss'], metric_logging_callback=metric_logging_callback)

        # Callback for sheduler
        if self.scheduler is not None:
            self.callbacks['scheduler'] = SchedulerCallback(self.scheduler, every_iteration=self.decay_every_iteration)

        # Callback for checkpointing
        if args.checkpoint_every:
            self.callbacks['checkpoint'] = CheckpointingCallback(checkpoint_folder=args.check_folder,
                                                                 checkpoint_every=args.checkpoint_every,
                                                                 model=self.model,
                                                                 optimizer=None if args.checkpoint_model_only else self.optimizer,
                                                                 scheduler=None if args.checkpoint_model_only else self.scheduler)

    def train_core(self, batch, args):

        # Set model in training mode
        self.model.train(True)

        # Compute loss
        loss = self.objective_fn(batch)

        # Backpropagation and update
        self.optimizer.zero_grad()
        loss.backward()
        if args.grad_clip_value is not None: torch.nn.utils.clip_grad_value_(self.model.parameters(), args.grad_clip_value)
        if args.grad_clip_norm is not None: torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clip_norm)
        self.optimizer.step()

        # Set model in eval mode
        self.model.train(False)

        return loss

    def train(self, args):

        # Begin training
        for callback in self.callbacks.values(): callback.on_train_begin(train_iterator=self.train_loader, num_epochs=args.num_epochs)

        # Loop over epochs
        for epoch in range(args.num_epochs):

            # Begin epoch
            for callback in self.callbacks.values(): callback.on_epoch_begin(train_iterator=self.train_loader, num_epochs=args.num_epochs, epoch=epoch)

            # Loop over batches
            for i, batch in enumerate(self.train_loader):

                # Begin batch
                for callback in self.callbacks.values(): callback.on_batch_begin(train_iterator=self.train_loader, num_epochs=args.num_epochs, epoch=epoch, iteration=i, batch=batch)

                # Perform core training operations
                loss = self.train_core(batch=batch, args=args)

                # End batch
                for callback in self.callbacks.values(): callback.on_batch_end(train_iterator=self.train_loader, num_epochs=args.num_epochs, epoch=epoch, iteration=i, batch=batch, loss=loss)

            # End epoch
            for callback in self.callbacks.values(): callback.on_epoch_end(train_iterator=self.train_loader, num_epochs=args.num_epochs, epoch=epoch)

        # End training
        for callback in self.callbacks.values(): callback.on_train_end(train_iterator=self.train_loader, num_epochs=args.num_epochs)

    def run(self, args, start_epoch=None):

        assert hasattr(self, "train_set"), "Need to register data with register_data(train_set, test_set)"
        assert hasattr(self, "model"), "Need to register model with register_model(model)"
        assert hasattr(self, "optimizer"), "Need to register optimizer with register_optimizer(optimizer)"

        if not self.env_initialized:
            self.prepare_env(args)

        print("Moving model to device...")
        self.model_to_device(args)

        print("Preparing data loaders...")
        self.prepare_data_loaders(args)

        print("Storing arguments...")
        self.store_arguments(args)

        print("Defining training...")
        self.define_training(args)

        print("Starting training...")
        self.train(args)
