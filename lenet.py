import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet3D(nn.Module):
    def __init__(self):
        super(LeNet3D, self).__init__()

        self.conv1 = nn.Conv3d(1, 6, kernel_size=(3, 3, 3))
        self.pool = nn.MaxPool3d(2, 2)
        self.conv2 = nn.Conv3d(6, 16, kernel_size=(3, 3, 3))
        self.fc1 = nn.Linear(1024, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        # print(x.shape)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # print(x.shape)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # print(x.shape)
        x = x.view(-1, 1024)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        # print(x.shape)
        x = F.relu(self.fc2(x))
        # print(x.shape)
        x = self.fc3(x)
        # print(x.shape)
        return x

    def intermediate(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # print(x.shape)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # print(x.shape)
        x = x.view(-1, 16 * 9 * 4 * 9)
        # print(x.shape)
        x = F.relu(self.fc1(x))

        return x