from .utils import FastLoader, DEFAULT_PATH
from torchvision.datasets import CIFAR10
from torchvision import transforms


class CategoricalCIFAR10(FastLoader):

    def __init__(self, root=DEFAULT_PATH, download=True):

        self.root = root

        # Define transformations
        transform = transforms.Compose([transforms.ToTensor(),
                                        lambda x: (255*x).floor().long()])

        # Load data
        self.train = CIFAR10(root, train=True, transform=transform, download=download)
        self.test = CIFAR10(root, train=False, transform=transform)
