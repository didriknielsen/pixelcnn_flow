from torch.utils.data import DataLoader
from os import path
import importlib
DEFAULT_PATH = path.join(path.dirname(path.dirname(importlib.util.find_spec("pixelflow").origin)), 'datasets') # [pixelcnn_flow/datasets/.]


class FastLoader():

    def get_data_loaders(self, batch_size, pin_memory=True, num_workers=4):
        train_iter = DataLoader(self.train, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers)
        test_iter = DataLoader(self.test, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)
        return train_iter, test_iter
