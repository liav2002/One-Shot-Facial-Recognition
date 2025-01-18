import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, Resize, ToTensor

from utils.config_loader import load_config
from data.pairs_dataset import PairsDataset

from model.siamese_network import SiameseNetwork


class Trainer:
    def __init__(self, config_path: str):
        """
        Initialize the Trainer with the configuration file.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        self.config = load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SiameseNetwork(self.config).to(self.device)

        self._prepare_data()
        self.loss_fn = self._regularized_loss
        self.optimizer = self._initialize_optimizer()

        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=self.config['training']['learning_rate_decay']
        )

        self.num_epochs = self.config['training']['num_epochs']
        self.early_stopping_patience = self.config['training']['early_stopping']['patience']
        self.early_stopping_delta = self.config['training']['early_stopping']['min_delta']

        self.checkpoint_dir = self.config['logging']['checkpoint_dir']

    def _get_transforms(self):
        """Define and return image transformations."""
        return Compose([
            Resize(self.config['data']['image_size']),
            ToTensor(),
        ])

    def _prepare_data(self):
        """Prepare DataLoaders for training and validation, ensuring people are exclusive to each set."""
        full_df = PairsDataset.load_pairs_from_txt_file(
            self.config['data']['train_pairs_path'],
            self.config['data']['lfw_data_path']
        )

        unique_people = pd.concat([full_df['person1'], full_df['person2']]).unique()

        train_people, val_people = train_test_split(
            unique_people, test_size=self.config['validation']['val_split'],
            shuffle=self.config['validation']['shuffle'],
            random_state=self.config['validation']['random_seed']
        )

        train_df = full_df[
            full_df['person1'].isin(train_people) & full_df['person2'].isin(train_people)
            ]

        val_df = full_df[
            full_df['person1'].isin(val_people) & full_df['person2'].isin(val_people)
            ]

        # Validate no people overlap between train and validation sets
        train_people_set = set(pd.concat([train_df['person1'], train_df['person2']]).unique())
        val_people_set = set(pd.concat([val_df['person1'], val_df['person2']]).unique())
        assert train_people_set.isdisjoint(val_people_set), "Train and validation sets have overlapping people!"

        transforms = self._get_transforms()

        self.train_loader = DataLoader(
            PairsDataset(train_df, transform=transforms),
            batch_size=self.config['training']['batch_size'],
            shuffle=True
        )
        self.val_loader = DataLoader(
            PairsDataset(val_df, transform=transforms),
            batch_size=self.config['training']['batch_size'],
            shuffle=False
        )

    def _initialize_optimizer(self):
        """Initialize the optimizer based on configuration."""
        optimizer_name = self.config['training']['optimizer']
        learning_rate = self.config['training']['learning_rate']
        momentum = self.config['training']['momentum']
        weight_decay = self.config['training']['weight_decay']

        if optimizer_name == "SGD":
            return optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_name == "Adam":
            return optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _regularized_loss(self, outputs, labels):
        """
        Custom regularized cross-entropy loss.

        Args:
            outputs (torch.Tensor): Predicted similarity scores.
            labels (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Loss value.
        """
        cross_entropy = labels * torch.log(outputs) + (1 - labels) * torch.log(1 - outputs)
        l2_reg = torch.tensor(0., requires_grad=True).to(self.device)
        for param in self.model.parameters():
            l2_reg += torch.norm(param, 2)
        return -cross_entropy.mean() + self.config['training']['weight_decay'] * l2_reg

    def train_one_epoch(self):
        """Perform one epoch of training."""
        self.model.train()
        running_loss = 0.0

        for batch in self.train_loader:
            (img1, img2), labels = batch
            img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(img1, img2)
            loss = self.loss_fn(outputs, labels.unsqueeze(1).float())

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        self.scheduler.step()

        return running_loss / len(self.train_loader)

    def validate(self):
        """Evaluate the model on the validation set."""
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                (img1, img2), labels = batch
                img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device)

                outputs = self.model(img1, img2)

                loss = self.loss_fn(outputs, labels.unsqueeze(1).float())
                val_loss += loss.item()

        return val_loss / len(self.val_loader)

    def save_model(self, path: str):
        """Save the model checkpoint."""
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str):
        """Load the model checkpoint."""
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)

    def train(self):
        """Train the model with validation and early stopping."""
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.num_epochs):
            train_loss = self.train_one_epoch()
            val_loss = self.validate()

            print(f"Epoch {epoch + 1}/{self.num_epochs}:")
            print(f"    Train Loss: {train_loss:.4f}")
            print(f"    Validation Loss: {val_loss:.4f}")

            if val_loss < best_val_loss - self.early_stopping_delta:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model(f"{self.checkpoint_dir}/best_model.pth")
                print("    Model improved. Saving...")
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    print("    Early stopping triggered.")
                    break
