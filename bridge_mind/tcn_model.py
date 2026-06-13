import torch
import torch.nn as nn

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=kernel//2)
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

class BridgeTCN(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.tcn = nn.Sequential(
            TCNBlock(input_size, 256),
            TCNBlock(256, 256),
            TCNBlock(256, 128)
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = x.permute(0, 2, 1)
        return self.classifier(x)
