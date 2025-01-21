import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import precision_score, recall_score, f1_score

from utils.logger import get_logger
from utils.load_pairs import load_pairs_from_txt_file
from utils.train_val_split import split_pairs_by_connected_components

from data.pairs_dataset import PairsDataset
from model.siamese_network import SiameseNetwork


class Trainer:
    def __init__(self, config: dict):
        self.logger = get_logger()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SiameseNetwork(self.config).to(self.device)
        self._prepare_data()

        self.loss_fn = self._initialize_loss_function()
        self.optimizer = self._initialize_optimizer()
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=self.config['training']['learning_rate_decay']
        )

        self.num_epochs = self.config['training']['num_epochs']
        self.early_stopping_patience = self.config['training']['early_stopping']['patience']
        self.early_stopping_delta = self.config['training']['early_stopping']['min_delta']
        self.checkpoint_dir = self.config['logging']['checkpoint_dir']
        self.best_val_loss = float('inf')
        self.start_epoch = 0

    def _get_transforms(self, stage: str):
        """
        Define and return image transformations based on the configuration.

        Args:
            stage (str): The stage of the data (e.g., "train", "val").

        Returns:
            torchvision.transforms.Compose: Composed transformations.
        """
        transform_list = []
        for transform_config in self.config['data']['transformations'][stage]:
            transform_type = transform_config['type']
            if transform_type == "Resize":
                transform_list.append(transforms.Resize(transform_config['size']))
            elif transform_type == "RandomHorizontalFlip":
                transform_list.append(transforms.RandomHorizontalFlip(p=transform_config['probability']))
            elif transform_type == "RandomRotation":
                transform_list.append(transforms.RandomRotation(degrees=transform_config['degrees']))
            elif transform_type == "RandomAdjustSharpness":
                transform_list.append(transforms.RandomAdjustSharpness(
                    sharpness_factor=transform_config['sharpness_factor'],
                    p=transform_config['probability']
                ))
            elif transform_type == "ToTensor":
                transform_list.append(transforms.ToTensor())
            else:
                raise ValueError(f"Unsupported transform type: {transform_type}")

        return transforms.Compose(transform_list)

    def _prepare_data(self):
        full_df = load_pairs_from_txt_file(
            self.config['data']['train_pairs_path'],
            self.config['data']['lfw_data_path']
        )

        train_df, val_df = split_pairs_by_connected_components(full_df, val_split=self.config["validation"]["val_split"],
                                                               random_seed=self.config["validation"]["random_seed"])

        train_transforms = self._get_transforms(stage="train")
        val_transforms = self._get_transforms(stage="val")

        self.train_loader = DataLoader(
            PairsDataset(train_df, transform=train_transforms),
            batch_size=self.config['training']['batch_size'],
            shuffle=True
        )

        self.val_loader = DataLoader(
            PairsDataset(val_df, transform=val_transforms),
            batch_size=self.config['training']['batch_size'],
            shuffle=False
        )

        self.logger.log_message(f"Data prepared: {len(train_df)} training pairs, {len(val_df)} validation pairs.")

    def _initialize_optimizer(self):
        optimizer_name = self.config['training']['optimizer']
        learning_rate = self.config['training']['learning_rate']
        momentum = self.config['training']['momentum']
        weight_decay = self.config['training']['weight_decay']
        if optimizer_name == "SGD":
            self.logger.log_message(
                f"SGD optimizer defined with lr={learning_rate}, momentum={momentum}, weight_decay={weight_decay}")
            return optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_name == "Adam":
            self.logger.log_message(
                f"Adam optimizer defined with lr={learning_rate}, weight_decay={weight_decay}")
            return optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _initialize_loss_function(self):
        loss_function_name = self.config['training']['loss_function']
        if loss_function_name == "BinaryCrossEntropy":
            self.logger.log_message(f"nn.BCELoss function defined as loss function.")
            return nn.BCELoss()
        elif loss_function_name == "RegularizedCrossEntropy":
            self.logger.log_message(f"_regularized_loss function defined as loss function.")
            return self._regularized_loss
        else:
            raise ValueError(f"Unsupported loss function: {loss_function_name}")

    def _regularized_loss(self, outputs, labels):
        cross_entropy = labels * torch.log(outputs) + (1 - labels) * torch.log(1 - outputs)
        l2_reg = torch.tensor(0., requires_grad=True).to(self.device)
        for param in self.model.parameters():
            l2_reg = l2_reg + torch.norm(param, 2)
        return -cross_entropy.mean() + self.config['training']['weight_decay'] * l2_reg

    def save_checkpoint(self, epoch, path):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": epoch,
            "best_val_loss": self.best_val_loss,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
        self.logger.log_message(f"Checkpoint saved at {path}")
        self.logger.log_artifacts(os.path.dirname(path))

    def load_checkpoint(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.start_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.logger.log_message(f"Checkpoint loaded from {path}. Resuming from epoch {self.start_epoch + 1}.")

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(self.train_loader, start=1):
            (img1, img2), labels = batch
            img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(img1, img2).squeeze()
            loss = self.loss_fn(outputs, labels.float())
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            predictions = (outputs > 0.5).long()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            if batch_idx % self.logger.log_interval == 0:
                accuracy = correct / total
                self.logger.log_message(
                    f"Epoch {epoch}, Batch {batch_idx}: Loss = {loss.item():.4f}, Accuracy = {accuracy:.4f}")

        self.scheduler.step()
        train_loss = running_loss / len(self.train_loader)
        train_accuracy = correct / total
        self.logger.log_metrics({
            "train_loss": train_loss,
            "train_accuracy": train_accuracy
        }, step=epoch)
        return train_loss

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0.0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for batch in self.val_loader:
                (img1, img2), labels = batch
                img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device)

                outputs = self.model(img1, img2).squeeze()
                loss = self.loss_fn(outputs, labels.float())
                val_loss += loss.item()

                predictions = (outputs > 0.5).long()
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())

        val_loss /= len(self.val_loader)
        accuracy = (np.array(all_predictions) == np.array(all_labels)).mean()
        precision = precision_score(all_labels, all_predictions, zero_division=1)
        recall = recall_score(all_labels, all_predictions, zero_division=1)
        f1 = f1_score(all_labels, all_predictions, zero_division=1)

        self.logger.log_message(
            f"Validation: Loss = {val_loss:.4f}, Accuracy = {accuracy:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}")
        self.logger.log_metrics({
            "val_loss": val_loss,
            "val_accuracy": accuracy,
            "val_precision": precision,
            "val_recall": recall,
            "val_f1": f1
        }, step=epoch)
        return val_loss

    def train(self):
        patience_counter = 0

        for epoch in range(self.start_epoch + 1, self.num_epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate(epoch)

            self.logger.log_message(
                f"Epoch {epoch}/{self.num_epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

            if val_loss < self.best_val_loss - self.early_stopping_delta:
                self.best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(epoch, f"{self.checkpoint_dir}/best_model.pth")
                self.logger.log_message(f"Model improved at epoch {epoch}. Checkpoint saved.")
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    self.logger.log_message("Early stopping triggered.")
                    break
