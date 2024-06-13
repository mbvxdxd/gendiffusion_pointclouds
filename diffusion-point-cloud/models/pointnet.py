import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(PointNetEncoder, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, latent_dim)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(latent_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.bn5(self.fc2(x))
        return x

class PointNetClassifier(nn.Module):
    def __init__(self, num_classes, latent_dim=256):
        super(PointNetClassifier, self).__init__()
        self.encoder = PointNetEncoder(latent_dim)
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

if __name__ == "__main__":
    model = PointNetClassifier(num_classes=10)
    print(model)
