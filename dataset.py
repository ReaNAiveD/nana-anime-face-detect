import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class CartoonDataset(Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, 'cartoon_train'))))
        self.boxes = {}
        with open(os.path.join(root, 'train.txt'), 'r') as t:
            for line in t.readlines():
                infos = line.split(',')
                if infos[0] not in self.boxes.keys():
                    self.boxes[infos[0]] = []
                self.boxes[infos[0]].append([int(axis) for axis in infos[1:]])

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, 'cartoon_train', self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        boxes = torch.as_tensor(self.boxes[self.imgs[idx]], dtype=torch.float32)
        labels = torch.ones((len(self.boxes[self.imgs[idx]]),), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(self.boxes[self.imgs[idx]]),), dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area, "iscrowd": iscrowd}

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.imgs)
