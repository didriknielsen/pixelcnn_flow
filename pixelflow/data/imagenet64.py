from .utils import FastLoader, DEFAULT_PATH
from .imagenet64_dataset import ImageNet64Dataset
from torchvision import transforms


class CategoricalImageNet64(FastLoader):

    def __init__(self, root=DEFAULT_PATH, download=True):

        self.root = root

        # Define transformations
        transform = transforms.Compose([transforms.ToTensor(),
                                        lambda x: (255*x).floor().long()])

        # Load data
        self.train = ImageNet64Dataset(root, train=True, transform=transform, download=download)
        self.test = ImageNet64Dataset(root, train=False, transform=transform)
