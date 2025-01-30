import torch
import torch.nn as nn

from src.utils.logger import get_logger


class SiameseNetwork(nn.Module):
    def __init__(self, config: dict, init_weights: bool = True):
        """
        Siamese Network architecture, dynamically built based on configuration.

        Args:
            config (dict): Configuration dictionary containing model parameters.
        """
        super(SiameseNetwork, self).__init__()

        self.logger = get_logger()
        self.logger.log_message("\n")

        cnn_config = config['cnn_layers']
        input_size = config['input_size']
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

        embedding_dim = config['fc_layers']['embedding_dim']
        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, embedding_dim),
            nn.Sigmoid(),
        )

        self.similarity = nn.Sequential(
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid(),
        )

        self.init_config = config['initialization']

        if init_weights:
            self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        """
        Initialize weights and biases for the network using configuration.

        Args:
            module (nn.Module): The layer to initialize.
        """
        method = self.init_config.get('method')

        if method is None:
            raise ValueError("Initialization method must be specified in the configuration!")

        if isinstance(module, nn.Conv2d):
            if method == 'xavier':
                nn.init.xavier_normal_(module.weight)
                self.logger.log_message(f"Initialized Conv2d weights with Xavier initialization.")
            elif method == 'normal':
                params = self.init_config.get('normal_params')
                if params is None:
                    raise ValueError("normal_params must be defined in the configuration for normal initialization!")

                nn.init.normal_(
                    module.weight,
                    mean=params['conv_weights']['mean'],
                    std=params['conv_weights']['std']
                )
                self.logger.log_message(
                    f"Initialized Conv2d weights with mean={params['conv_weights']['mean']}, "
                    f"std={params['conv_weights']['std']}."
                )
                if module.bias is not None:
                    nn.init.normal_(
                        module.bias,
                        mean=params['conv_biases']['mean'],
                        std=params['conv_biases']['std']
                    )
                    self.logger.log_message(
                        f"Initialized Conv2d bias with mean={params['conv_biases']['mean']}, "
                        f"std={params['conv_biases']['std']}."
                    )
            else:
                raise ValueError(f"Unsupported initialization method: {method}")

        elif isinstance(module, nn.Linear):
            if method == 'xavier':
                nn.init.xavier_normal_(module.weight)
                self.logger.log_message(f"Initialized Linear weights with Xavier initialization.")
            elif method == 'normal':
                params = self.init_config.get('normal_params')
                if params is None:
                    raise ValueError("normal_params must be defined in the configuration for normal initialization!")

                nn.init.normal_(
                    module.weight,
                    mean=params['fc_weights']['mean'],
                    std=params['fc_weights']['std']
                )
                self.logger.log_message(
                    f"Initialized Linear weights with mean={params['fc_weights']['mean']}, "
                    f"std={params['fc_weights']['std']}."
                )
                if module.bias is not None:
                    nn.init.normal_(
                        module.bias,
                        mean=params['fc_biases']['mean'],
                        std=params['fc_biases']['std']
                    )
                    self.logger.log_message(
                        f"Initialized Linear bias with mean={params['fc_biases']['mean']}, "
                        f"std={params['fc_biases']['std']}."
                    )
            else:
                raise ValueError(f"Unsupported initialization method: {method}")

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
