import random

from torchvision.transforms import functional as F
from PIL import Image


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, img, target):
        if random.random() < self.prob:
            height, width = img.shape[-2:]
            img = img.flip(-1)
            bbox = target["boxes"]
            bbox[:, [2, 0]] = width - bbox[:, [0, 2]]
            target["boxes"] = bbox
        return img, target


class ToTensor(object):
    def __call__(self, img, target):
        img = F.to_tensor(img)
        return img, target


class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, target):
        width, height = img.size
        img = F.resize(img, self.size, self.interpolation)
        new_width, new_height = img.size
        bbox = target["boxes"]
        bbox[:, [0, 2]] = new_width * bbox[:, [0, 2]] / width
        bbox[:, [1, 3]] = new_height * bbox[:, [1, 3]] / height
        area = (bbox[:, 3] - bbox[:, 1]) * (bbox[:, 2] - bbox[:, 0])
        target["boxes"] = bbox
        target["area"] = area
        return img, target