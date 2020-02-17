from .utils import FastLoader, DEFAULT_PATH
from .imagenet32_dataset import ImageNet32Dataset
from torchvision import transforms


class CategoricalImageNet32(FastLoader):

    def __init__(self, root=DEFAULT_PATH, download=True):

        self.root = root

        # Define transformations
        transform = transforms.Compose([transforms.ToTensor(),
                                        lambda x: (255*x).floor().long()])

        # Load data
        self.train = ImageNet32Dataset(root, train=True, transform=transform, download=download)
        self.test = ImageNet32Dataset(root, train=False, transform=transform)
