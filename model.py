import torch
import torch.nn as nn


class LipRead3D(nn.Module):
    def __init__(self, num_classes=13):
        super(LipRead3D, self).__init__()
        # 3D Convolutional Blocks for Spatiotemporal extraction
        self.conv1 = nn.Conv3d(1, 32, kernel_size=(
            3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(
            3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2))
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv3 = nn.Conv3d(
            64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # Sequence modeling
        self.fc1 = nn.Linear(64 * 5 * 7, 256)
        self.dropout = nn.Dropout(0.5)
        self.gru = nn.GRU(256, 128, num_layers=2,
                          batch_first=True, bidirectional=True, dropout=0.3)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # x input shape: (Batch, Frames, Height, Width)
        x = x.unsqueeze(1)  # Add channel dimension

        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.relu(self.conv3(x))
        x = self.pool3(x)

        B, C, T, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B, T, -1)

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x, _ = self.gru(x)

        # Grab final time step
        x = x[:, -1, :]
        out = self.fc2(x)
        return out
