       

import torch
import torch.nn as nn

class ConvEncoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 8, 5, stride=2, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Conv2d(8, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc = nn.Linear(64, latent_dim)

    def forward(self, x):
        # x: (B, 1, H, W)
        h = self.net(x)
        h = h.view(h.size(0), -1)
        z = self.fc(h)
        return z


class SiameseShearNet(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = ConvEncoder(1, latent_dim)
        self.head = nn.Sequential(
            nn.Linear(2 * latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )

    def forward(self, x):
        
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]

        z1 = self.encoder(x1)
        z2 = self.encoder(x2)

        z = torch.cat([z1, z2], dim=1)
        return self.head(z).squeeze(1)


class MLPNet(nn.Module):
    def __init__(self, H, W):
        super().__init__()
        self.H = H
        self.W = W
        self.MLP = nn.Sequential(
            nn.Linear(self.H* self.W, 1000),
            nn.ReLU(),

            nn.Linear(1000, 500),
            nn.ReLU(),

            nn.Linear(500, 80),
            nn.ReLU(),

            nn.Linear(80, 16),
            nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        # x: (B, H, W)
        x = x.flatten(1)
        out = self.MLP(x)
        return out.squeeze(-1)