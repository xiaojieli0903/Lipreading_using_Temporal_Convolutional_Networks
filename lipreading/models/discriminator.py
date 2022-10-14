import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, input_dim, feature_dim=512, dropout_rate=0.5):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, feature_dim * 2),
            nn.ReLU(False),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(False))
        self.cls = nn.Linear(feature_dim, 2)

    def forward(self, x):
        x = self.fc(x.view(x.shape[0], -1))
        x = self.cls(x)
        return x
