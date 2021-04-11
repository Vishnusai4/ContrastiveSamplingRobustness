import numpy as np
from torchvision import transforms
np.random.seed(0)


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.transform1 = base_transform
        self.transform2 = transforms.Compose([transforms.ToTensor()])
        self.n_views = n_views

    def __call__(self, x):
        return [self.transform1(x),self.transform2(x)]
