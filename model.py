import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F

class EmbeddingNet(nn.Module):
    """Feature extractor using a pretrained ResNet50 backbone."""
    def __init__(self, pretrained=True, embedding_size=256):
        super().__init__()
        resnet = models.resnet50(pretrained=pretrained)
        # Remove the final classifier
        modules = list(resnet.children())[:-1]  # remove fc
        self.backbone = nn.Sequential(*modules)
        in_features = resnet.fc.in_features
        # projection head
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_size)
        )

    def forward(self, x):
        x = self.backbone(x)  # shape: (B, C, 1, 1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)  # L2 normalize embeddings
        return x

class SiameseNet(nn.Module):
    """Siamese network that returns embeddings for pairwise comparison."""
    def __init__(self, embedding_net):
        super().__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        e1 = self.embedding_net(x1)
        e2 = self.embedding_net(x2)
        return e1, e2

def contrastive_loss(e1, e2, label, margin=1.0):
    # label: 1 if similar, 0 if dissimilar
    distances = (e1 - e2).pow(2).sum(1)  # squared euclidean
    losses_similar = label * distances
    losses_dissimilar = (1 - label) * F.relu(margin - (distances + 1e-8).sqrt()).pow(2)
    loss = 0.5 * (losses_similar + losses_dissimilar).mean()
    return loss
