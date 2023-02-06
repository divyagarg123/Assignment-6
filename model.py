import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, stride=1, bias=False),   # r_in = 1, jin = 1 jout=1*1 =1, k=3, rout = 1+(3-1)*1 = 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0, stride=1, dilation=2, bias=False),# r_in = 3, jin = 1 jout=1*1 =1, k=5, rout = 3+(5-1)*1 = 7
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.BatchNorm2d(64)
        )
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=0, stride=1, bias=False),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), padding=0, stride=1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.BatchNorm2d(128)
        )
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=0, stride=1, bias=False),
            # r_in = 21, jin = 8 jout=8*1 =8, k=1, rout = 5+(1-1)*8 = 21
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.BatchNorm2d(128)
        )
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), padding=0, stride=1, bias=False),
            # r_in = 21, jin = 8 jout=8*2 =8, k=3, rout = 21+(3-1)*8 = 37
        )
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=24))  # r_in = 37, jin = 8,s=2, jout=8*2 =16, k=2, rout = 37+(2-1)*8 = 45
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        # print(x.shape)
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        #print(x.shape)
        x = self.gap(x)
        #print(x.shape)
        x = x.view(-1, 256)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1, 10)
        x = F.log_softmax(x, dim=-1)
        return x