import torch
from torch import nn
class ECGCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(nn.Conv1d(1,16,9,padding=4),nn.BatchNorm1d(16),nn.ReLU(),nn.MaxPool1d(2),nn.Conv1d(16,32,7,padding=3),nn.BatchNorm1d(32),nn.ReLU(),nn.MaxPool1d(2),nn.Conv1d(32,64,5,padding=2),nn.BatchNorm1d(64),nn.ReLU(),nn.AdaptiveAvgPool1d(1))
        self.classifier=nn.Sequential(nn.Flatten(),nn.Dropout(.25),nn.Linear(64,2))
    def forward(self,x): return self.classifier(self.features(x))
