import math
import sys
import time
import os

import torch
import torchvision
from dataset import CartoonDataset
from torch.utils.data import DataLoader
import transforms as T
from model import faster_rcnn
import matplotlib.pyplot as plt
import numpy as np

imsize = 512
classes_num = 2


def get_transform(train):
    transforms = []
    transforms.append(T.Resize((imsize, imsize)))
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def collate_fn(batch):
    return tuple(zip(*batch))


# dataset = CartoonDataset('cartoon_dataset', get_transform(False))
# data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)
def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def area(box):
    return (box[3] - box[1]) * (box[2] - box[0])


def cal_iou(box_a, box_b):
    inter_x_max = min(box_a[2], box_b[2])
    inter_x_min = max(box_a[0], box_b[0])
    inter_y_max = min(box_a[3], box_b[3])
    inter_y_min = max(box_a[1], box_b[1])
    inter_area = ((inter_x_max - inter_x_min) if inter_x_max > inter_x_min else 0) * \
                 ((inter_y_max - inter_y_min) if inter_y_max > inter_y_min else 0)
    out_area = area(box_a) + area(box_b) - inter_area
    return inter_area / out_area


def draw_pr(precision, recall, target_dir='plt'):
    plt.figure(figsize=(8, 8))
    plt.plot(recall.numpy(), precision.numpy())
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(target_dir, 'pr.jpg'))


def prepare_data_loader(root='cartoon_dataset', test_size=50):
    dataset = CartoonDataset(root, get_transform(True))
    indices = torch.randperm(len(dataset)).tolist()
    data_loader = None
    if test_size < len(dataset):
        dataset_train = torch.utils.data.Subset(dataset, indices[:-test_size])
        data_loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=2, shuffle=True, num_workers=4,
            collate_fn=collate_fn)

    data_loader_test = None
    if test_size > 0:
        dataset_test = torch.utils.data.Subset(CartoonDataset(root, get_transform(False)), indices[-test_size:])
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, shuffle=False, num_workers=4,
            collate_fn=collate_fn)
    return data_loader, data_loader_test


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in iter(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        loss_value = losses.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            # print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    model.to(device)
    threshold = torch.as_tensor(
        [.9999, .9995, .999, .998, .997, .996, .995, .99, 0.98, 0.97, .96, 0.95, .94, .93, .92, .91, .90, .89, .88, .87, .86, .85, .84, .83, .82, .81, .80, .75, .7, .65, .6, .58, .55, .52, .5, .45, .4, .35, .3, .2, .1, .05, .01],
        dtype=torch.float32)
    true_pos = torch.as_tensor([0 for i in range(threshold.shape[0])], dtype=torch.int32)
    false_pos = torch.as_tensor([0 for i in range(threshold.shape[0])], dtype=torch.int32)
    recall_found = torch.as_tensor([0 for i in range(threshold.shape[0])], dtype=torch.int32)
    total = 0
    for images, targets in iter(data_loader):
        found = [0 for i in range(targets[0]['boxes'].shape[0])]
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        total = total + targets[0]['boxes'].shape[0]
        result = model(images)
        for idx in range(result[0]['scores'].shape[0]):
            score = result[0]['scores'][idx]
            max_iou = 0
            max_iou_idx = 0
            for gt_idx in range(targets[0]['boxes'].shape[0]):
                iou = cal_iou(result[0]['boxes'][idx, :], targets[0]['boxes'][gt_idx, :])
                if iou > max_iou:
                    max_iou = iou
                    max_iou_idx = gt_idx
                max_iou = max(max_iou, iou)
            if max_iou > 0.7:
                true_pos[:] = true_pos[:] + threshold.le(score)
                found[max_iou_idx] = max(found[max_iou_idx], score)
            else:
                false_pos[:] = false_pos[:] + threshold.le(score)
        for idx in range(threshold.shape[0]):
            recall_found[idx] = \
                recall_found[idx].item() + sum([1 if score >= threshold[idx].item() else 0 for score in found])
        a = 1
    precision = true_pos / (true_pos + false_pos + 0.00000000001)
    recall = recall_found.float() / total
    f_measure = 2 / (1 / precision + 1 / recall)
    print('threshold: ', threshold.data)
    print('precision: ', precision.data)
    print('recall: ', recall.data)
    print('f_measure: ', f_measure.data)
    draw_pr(precision, recall)


def train(model, device, root='cartoon_dataset', model_target_dir='model',
          num_epochs=10, test_size=50):
    data_loader, data_loader_test = prepare_data_loader(root, test_size)

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer=optimizer, data_loader=data_loader, device=device, epoch=epoch)
        lr_scheduler.step()
        if test_size > 0:
            evaluate(model, data_loader=data_loader_test, device=device)
    torch.save(model.state_dict(), os.path.join(model_target_dir, 'model.pth'))


def evaluate_model(model, device, root='cartoon_dataset', test_size=50):
    _, data_loader_test = prepare_data_loader(root, test_size)

    model.to(device)
    if test_size > 0:
        evaluate(model, data_loader=data_loader_test, device=device)


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = faster_rcnn(2, True)
    train(model, device, root='cartoon_dataset', num_epochs=10)
    # evaluate_model(model, device, test_size=50)
