from model import faster_rcnn
import torch
from train import train, evaluate_model
from predict import predict, detect_image


model = faster_rcnn(pretrained=True)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


if __name__ == '__main__':
    # train(model, device, num_epochs=1)
    # evaluate_model(model, device, test_size=2000)

    # predict(model, device)
    detect_image(model, device, 'cartoon_dataset/cartoon_test/009513.jpg')
