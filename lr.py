import torch
import torch.nn as nn
import torch.nn.functional as F


class LogisticRegression(nn.Module):
    def __init__(self, in_dim, n_class):
        super(LogisticRegression, self).__init__()
        self.fc1 = nn.Linear(in_dim,64)
        self.fc2 = nn.Linear(64,n_class)
        #self.logistic = nn.Linear(in_dim, n_class)

    def forward(self, x):
        x = x.view(x.size(0), -1).contiguous()
        x = self.fc1(x)
        x = self.fc2(x)
        out = F.log_softmax(x, dim=1)
        return out
    
    def intermediate(self, x):
        x = x.view(x.size(0), -1).contiguous()
        out = self.fc1(x)
        return out


def LROnMedical():
    return LogisticRegression(120, 3)
