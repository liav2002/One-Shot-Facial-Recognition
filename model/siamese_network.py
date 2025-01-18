import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseNetwork(nn.Module):
    def __init__(self):
        """
        Siamese Network architecture as described in the paper.
        """
        super(SiameseNetwork, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=10, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 128, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=4, stride=1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.Sigmoid(),
        )

        self.similarity = nn.Sequential(
            nn.Linear(4096, 1),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2):
        """
        Forward pass for the Siamese Network.

        Args:
            x1 (torch.Tensor): First input image tensor of shape [B, 1, H, W].
            x2 (torch.Tensor): Second input image tensor of shape [B, 1, H, W].

        Returns:
            torch.Tensor: Similarity score between the two inputs, shape [B, 1].
        """
        # Shared CNN feature extraction
        embedding1 = self.cnn(x1)
        embedding2 = self.cnn(x2)

        # Flatten the embeddings
        embedding1 = embedding1.view(embedding1.size(0), -1)
        embedding2 = embedding2.view(embedding2.size(0), -1)

        # Process through fully connected layers
        embedding1 = self.fc(embedding1)
        embedding2 = self.fc(embedding2)

        # Compute the absolute difference
        diff = torch.abs(embedding1 - embedding2)

        # Compute similarity score
        similarity_score = self.similarity(diff)

        return similarity_score
