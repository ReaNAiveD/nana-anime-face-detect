import os
import torch
from train import faster_rcnn
from PIL import Image, ImageDraw
import torchvision.transforms as T
import numpy as np


def get_transforms():
    return T.Compose([T.ToTensor()])


def predict(model, device, root='cartoon_dataset/cartoon_test', result_dir='result', score_threshold=0.89):
    imgs = list(sorted(os.listdir(root)))
    model.eval()
    model.to(device)
    with open(os.path.join(result_dir, 'test_result.txt'), 'w') as f:
        f.truncate()
    for img_name in imgs:
        image = Image.open(os.path.join(root, img_name))
        image = image.convert('RGB')
        width, height = image.size
        transform = get_transforms()
        image = transform(image).to(device)
        result = model([image])[0]
        for idx in range(result['scores'].shape[0]):
            if result['scores'][idx].item() >= score_threshold:
                bbox = result['boxes'][idx]
                bottom, left, right, top = bbox_to_int(bbox, height, width)
                with open(os.path.join(result_dir, 'test_result.txt'), 'a') as f:
                    f.write('{},{},{},{},{}\n'.format(img_name, left, top, right, bottom))
                    f.flush()


def detect_image(model, device, image_path, output_dir='result', score_threshold=0.89):
    model.eval()
    model.to(device)
    raw_image = Image.open(image_path).convert('RGB')
    width, height = raw_image.size

    transform = get_transforms()
    image = transform(raw_image).to(device)
    result = model([image])[0]

    draw = ImageDraw.Draw(raw_image)
    for idx in range(result['scores'].shape[0]):
        if result['scores'][idx].item() >= score_threshold:
            bbox = result['boxes'][idx]
            bottom, left, right, top = bbox_to_int(bbox, height, width)
            draw.rectangle([left, top, right, bottom], outline=(255, 127, 127))
    del draw
    raw_image.save(os.path.join(output_dir, image_path.split('/')[-1]), quality=95)


def bbox_to_int(bbox, height, width):
    top = max(0, np.floor(bbox[1].item() + 0.5).astype('int32'))
    left = max(0, np.floor(bbox[0].item() + 0.5).astype('int32'))
    bottom = min(height, np.floor(bbox[3].item() + 0.5).astype('int32'))
    right = min(width, np.floor(bbox[2].item() + 0.5).astype('int32'))
    return bottom, left, right, top


if __name__ == '__main__':
    model = faster_rcnn(2, pretrained=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # predict(model, device)
    detect_image(model, device, 'cartoon_dataset/cartoon_test/008017.jpg')
