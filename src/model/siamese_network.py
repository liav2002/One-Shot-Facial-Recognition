import torch
import torch.nn as nn

from src.utils.logger import get_logger


class SiameseNetwork(nn.Module):
    def __init__(self, config: dict):
        """
        Siamese Network architecture, dynamically built based on configuration.

        Args:
            config (dict): Configuration dictionary containing model parameters.
        """
        super(SiameseNetwork, self).__init__()

        self.logger = get_logger()

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

        self.init_config = config['model']['initialization']

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        """
        Initialize weights and biases for the network using configuration.

        Args:
            module (nn.Module): The layer to initialize.
        """
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(
                module.weight,
                mean=self.init_config['conv_weights']['mean'],
                std=self.init_config['conv_weights']['std']
            )
            self.logger.log_message(
                f"Initialized Conv2d weights with mean={self.init_config['conv_weights']['mean']}, "
                f"std={self.init_config['conv_weights']['std']}")
            if module.bias is not None:
                nn.init.normal_(
                    module.bias,
                    mean=self.init_config['conv_biases']['mean'],
                    std=self.init_config['conv_biases']['std']
                )
                self.logger.log_message(
                    f"Initialized Conv2d bias with mean={self.init_config['conv_biases']['mean']}, "
                    f"std={self.init_config['conv_biases']['std']}")
        elif isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=self.init_config['fc_weights']['mean'],
                std=self.init_config['fc_weights']['std']
            )
            self.logger.log_message(
                f"Initialized Linear weights with mean={self.init_config['fc_weights']['mean']}, "
                f"std={self.init_config['fc_weights']['std']}")
            if module.bias is not None:
                nn.init.normal_(
                    module.bias,
                    mean=self.init_config['fc_biases']['mean'],
                    std=self.init_config['fc_biases']['std']
                )
                self.logger.log_message(
                    f"Initialized Linear bias with mean={self.init_config['fc_biases']['mean']}, "
                    f"std={self.init_config['fc_biases']['std']}")

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
