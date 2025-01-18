import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseNetwork(nn.Module):
    def __init__(self, config: dict):
        """
        Siamese Network architecture, dynamically built based on configuration.

        Args:
            config (dict): Configuration dictionary containing model parameters.
        """
        super(SiameseNetwork, self).__init__()

        cnn_config = config['model']['cnn_layers']
        input_size = config['model']['input_size']

        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=input_size[0],
                out_channels=cnn_config['conv1']['out_channels'],
                kernel_size=cnn_config['conv1']['kernel_size'],
                stride=cnn_config['conv1']['stride']
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=cnn_config['conv1']['pool_size']),

            nn.Conv2d(
                in_channels=cnn_config['conv1']['out_channels'],
                out_channels=cnn_config['conv2']['out_channels'],
                kernel_size=cnn_config['conv2']['kernel_size'],
                stride=cnn_config['conv2']['stride']
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=cnn_config['conv2']['pool_size']),

            nn.Conv2d(
                in_channels=cnn_config['conv2']['out_channels'],
                out_channels=cnn_config['conv3']['out_channels'],
                kernel_size=cnn_config['conv3']['kernel_size'],
                stride=cnn_config['conv3']['stride']
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=cnn_config['conv3']['pool_size']),

            nn.Conv2d(
                in_channels=cnn_config['conv3']['out_channels'],
                out_channels=cnn_config['conv4']['out_channels'],
                kernel_size=cnn_config['conv4']['kernel_size'],
                stride=cnn_config['conv4']['stride']
            ),
            nn.ReLU(),
        )

        self.flattened_size = cnn_config['conv4']['out_channels'] * 6 * 6

        embedding_dim = config['model']['fc_layers']['embedding_dim']
        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, embedding_dim),
            nn.Sigmoid(),
        )

        self.similarity = nn.Sequential(
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2):
        """
        Forward pass for the Siamese Network.

        Args:
            x1 (torch.Tensor): First input image tensor of shape [B, C, H, W].
            x2 (torch.Tensor): Second input image tensor of shape [B, C, H, W].

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
