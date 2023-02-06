from model import Model
from train import Train
from test import Test
import torch
import torch.optim as optim
from data_loader import DataLoader
import utils as ut

train_loader, test_loader = DataLoader().return_loaders()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
EPOCHS = 20


def train_model():
    model = Model().to(device)
    ut.print_model_summary(model)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    train = Train()
    test = Test()
    for epoch in range(EPOCHS):
        print("EPOCH:", epoch)
        train.train_model(model, device, train_loader, optimizer, epoch)
        test_losses, test_acc = test.test_model(model, device, test_loader)
    return test_losses, test_acc





bn_test_losses, bn_test_acc = train_model()

