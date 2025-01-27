import os
import torch
import optuna
import warnings
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import Callable, Tuple

from src.utils.logger import get_logger
from src.utils.load_pairs import load_pairs_from_txt_file
from src.utils.parser import parse_search_space, parse_hyperparameters
from src.utils.train_val_split import split_pairs_by_connected_components

from data.pairs_dataset import PairsDataset
from src.model.siamese_network import SiameseNetwork


class Trainer:
    def __init__(self, config: dict):
        """
        Initialize the Trainer class.

        The constructor sets up the necessary components for training, including the model,
        loss function, optimizer, scheduler, and data loaders. It also configures training
        parameters, logging, and early stopping.

        Args:
            config (dict): A dictionary containing the configuration for the training process.
                - 'training': Training parameters (e.g., learning rate, number of epochs, early stopping).
                - 'data': Data-related configurations (e.g., paths to datasets, transformations).
                - 'logging': Logging configurations (e.g., checkpoint directory, log intervals).
        """
        self.config = config
        self._validate_config()
        self.logger = get_logger()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_epochs = self.config['training']['num_epochs']
        self.early_stopping_patience = int(self.config['training']['early_stopping']['patience'])
        self.early_stopping_delta = float(self.config['training']['early_stopping']['min_delta'])
        self.checkpoint_dir = self.config['logging']['checkpoint_dir']
        self.best_val_accuracy = 0
        self.start_epoch = 0
        self.log_to_mlflow = False

    def _update_attributes_from_config(self):
        """
        Update class attributes based on the current configuration.

        This method ensures that the model, optimizer, loss function, scheduler,
        and data loaders are consistent with the current state of `self.config`.
        """
        # Initialize model
        self.model = SiameseNetwork(self.config).to(self.device)

        # Prepare data loaders
        self._prepare_data()

        # Initialize loss function
        self.loss_fn = self._initialize_loss_function()

        # Initialize optimizer
        self.optimizer = self._initialize_optimizer()

        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=self.config['training']['learning_rate_decay']
        )

        self.logger.log_message("Trainer attributes updated from configuration.")

    def _validate_config(self):
        """
        Validate that the required configuration keys are present.
        """
        required_keys = [
            'training.hyperparameter_tuning.enabled',
            'training.hyperparameter_tuning.method',
            'training.hyperparameter_tuning.num_trials',
            'training.hyperparameter_tuning.search_space',
            'training.num_epochs',
            'training.optimizer',
            'training.optimizer_params',
            'training.loss_function',
            'training.early_stopping'
        ]

        for key in required_keys:
            keys = key.split(".")
            value = self.config
            for k in keys:
                if k not in value:
                    raise ValueError(f"Missing required configuration key: {key}")
                value = value[k]

    def _get_transforms(self, stage: str) -> transforms.Compose:
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
            transform_params = {k: v for k, v in transform_config.items() if k != 'type'}
            try:
                transform_class = getattr(transforms, transform_type)
                transform_list.append(transform_class(**transform_params))
            except AttributeError:
                raise ValueError(f"Unsupported transform type: {transform_type}")

        transform_list.append(transforms.ToTensor())

        return transforms.Compose(transform_list)

    def _prepare_data(self) -> None:
        """
        Prepare the data for training and validation.

        This function loads the image pairs from the specified text file, splits the pairs into training
        and validation sets, applies the appropriate transformations, and initializes the data loaders
        for both sets.
        """
        full_df = load_pairs_from_txt_file(
            self.config['data']['train_pairs_path'],
            self.config['data']['lfw_data_path']
        )

        train_df, val_df = split_pairs_by_connected_components(full_df,
                                                               val_split=self.config["validation"]["val_split"],
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

        self.logger.log_message(
            f"Data prepared: {len(train_df)} training pairs, {len(val_df)} validation pairs "
            f"(Batch Size = {self.config['training']['batch_size']}).")

    def _initialize_optimizer(self) -> Optimizer:
        """
        Dynamically initialize and return the optimizer based on the configuration.

        This function reads the optimizer type and dynamically sets up the specified optimizer
        with only the parameters relevant to it.

        Returns:
            Optimizer: An instance of the specified optimizer.

        Raises:
            ValueError: If the specified optimizer is not found in `torch.optim`.
        """
        optimizer_name = self.config['training']['optimizer']
        optimizer_params = self.config['training'].get('optimizer_params', {})

        # Mapping of optimizers to their expected parameters
        valid_params = {
            "SGD": ["lr", "momentum", "weight_decay", "nesterov"],
            "Adam": ["lr", "betas", "eps", "weight_decay", "amsgrad"],
            "RMSprop": ["lr", "momentum", "alpha", "eps", "weight_decay", "centered"],
        }

        try:
            optimizer_class = getattr(optim, optimizer_name)
            if optimizer_name not in valid_params:
                raise ValueError(f"Unsupported optimizer: {optimizer_name}")

            # Filter optimizer_params to only include valid parameters for the chosen optimizer
            filtered_params = {key: optimizer_params[key] for key in valid_params[optimizer_name] if
                               key in optimizer_params}

            # Ensure numeric parameters are correctly typed
            if "weight_decay" in filtered_params:
                filtered_params["weight_decay"] = float(filtered_params["weight_decay"])
            if "lr" in filtered_params:
                filtered_params["lr"] = float(filtered_params["lr"])
            if "momentum" in filtered_params:
                filtered_params["momentum"] = float(filtered_params["momentum"])
            if "betas" in filtered_params:
                filtered_params["betas"] = tuple(filtered_params["betas"])

            self.logger.log_message(f"{optimizer_name} optimizer initialized with parameters: {filtered_params}")
            return optimizer_class(self.model.parameters(), **filtered_params)

        except AttributeError:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        except TypeError as e:
            raise ValueError(f"Error initializing optimizer {optimizer_name}: {e}")

    def _initialize_loss_function(self) -> Callable:
        """
        Initialize and return the loss function based on the configuration.

        This function sets up the loss function for training based on the specified name
        in the configuration. Supported loss functions include `BinaryCrossEntropy` and
        `RegularizedCrossEntropy`.

        Returns:
            Callable: A PyTorch loss function (e.g., `nn.BCELoss`) or a custom-defined callable.

        Raises:
            ValueError: If the loss function specified in the configuration is not supported.
        """
        loss_function_name = self.config['training']['loss_function']
        if loss_function_name == "BinaryCrossEntropy":
            self.logger.log_message(f"nn.BCELoss function defined as loss function.")
            return nn.BCELoss()
        elif loss_function_name == "RegularizedCrossEntropy":
            self.logger.log_message(f"_regularized_loss function defined as loss function.")
            return self._regularized_loss
        else:
            raise ValueError(f"Unsupported loss function: {loss_function_name}")

    def _regularized_loss(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the regularized loss for the model.

        The loss is calculated as the negative mean of the binary cross-entropy
        between `outputs` and `labels`, with an additional L2 regularization term
        applied to the model parameters.

        Args:
            outputs (torch.Tensor): The predicted outputs from the model, expected to be
                probabilities (e.g., from a sigmoid activation).
            labels (torch.Tensor): The ground truth labels for the outputs, with values
                in the range [0, 1].

        Returns:
            torch.Tensor: The computed regularized loss.
        """
        cross_entropy = labels * torch.log(outputs) + (1 - labels) * torch.log(1 - outputs)
        l2_reg = torch.tensor(0., requires_grad=True).to(self.device)
        for param in self.model.parameters():
            l2_reg = l2_reg + torch.norm(param, 2)

        weight_decay = float(self.config['training']['optimizer_params']['weight_decay'])

        return -cross_entropy.mean() + weight_decay * l2_reg

    def save_checkpoint(self, epoch: int, path: str) -> None:
        """
        Save the model checkpoint to the specified path.

        The checkpoint includes the model's state dictionary, optimizer state dictionary,
        scheduler state dictionary, the current epoch, and the best validation loss so far.
        The function ensures the directory structure exists before saving and logs the process.

        Args:
            epoch (int): The current epoch number.
            path (str): The file path where the checkpoint will be saved.

        Returns:
            None

        Raises:
            OSError: If the checkpoint cannot be saved due to an OS-level issue.
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": epoch,
            "best_val_accuracy": self.best_val_accuracy,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
        self.logger.log_message(f"Checkpoint saved at {path}")
        self.logger.log_artifacts(os.path.dirname(path))

    def load_checkpoint(self, path: str) -> None:
        """
        Load the model checkpoint from the specified path.

        This function restores the model's state, optimizer's state, scheduler's state,
        the starting epoch, and the best validation loss from a previously saved checkpoint.

        Args:
            path (str): The file path to the checkpoint file.

        Returns:
            None

        Raises:
            FileNotFoundError: If the specified checkpoint file does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.start_epoch = checkpoint["epoch"]
        self.best_val_accuracy = checkpoint["best_val_accuracy"]
        self.logger.log_message(f"Checkpoint loaded from {path}. Resuming from epoch {self.start_epoch + 1}.")

    def run_bayesian_search(self):
        """
        Run Bayesian optimization for hyperparameter tuning.
        """
        search_space = parse_search_space(self.config['training']['hyperparameter_tuning']['search_space'])
        half_epochs = max(1, self.num_epochs // 2)

        def objective(trial):
            val_loss = float('inf')

            params = {
                "batch_size": trial.suggest_int("batch_size", *search_space['batch_size']),
                "learning_rate": trial.suggest_float("learning_rate", *search_space['learning_rate'], log=True),
                "weight_decay": trial.suggest_float("weight_decay", *search_space['weight_decay'], log=True),
                "betas": trial.suggest_categorical("betas", search_space['betas']),
                "optimizer": trial.suggest_categorical("optimizer", search_space['optimizer']),
                "loss_function": trial.suggest_categorical("loss_function", search_space['loss_function'])
            }

            parsed_params = parse_hyperparameters(params)

            self.config['training']['batch_size'] = parsed_params["batch_size"]
            self.config['training']['optimizer_params']['lr'] = parsed_params["lr"]
            self.config['training']['optimizer_params']['weight_decay'] = parsed_params["weight_decay"]
            self.config['training']['optimizer_params']['betas'] = parsed_params["betas"]
            self.config['training']['optimizer'] = parsed_params["optimizer"]
            self.config['training']['loss_function'] = parsed_params["loss_function"]

            # Update trainer attributes based on the new config
            self._update_attributes_from_config()

            # Implement early stopping for Bayesian search
            best_val_accuracy = 0
            patience_counter = 0

            for epoch in range(1, half_epochs + 1):
                self.train_one_epoch(epoch)
                val_loss, val_accuracy = self.validate(epoch)

                # Check early stopping condition
                if val_accuracy > best_val_accuracy + self.early_stopping_delta:
                    best_val_accuracy = val_accuracy
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        self.logger.log_message("Early stopping triggered during Bayesian search.")
                        break

            return best_val_accuracy

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            # noinspection PyArgumentList
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=self.config['training']['hyperparameter_tuning']['num_trials'])

        best_params = study.best_params
        self.logger.log_message(f"Best hyperparameters found: {best_params}")

        # Update configuration with the best parameters
        self.config['training']['batch_size'] = best_params['batch_size']
        self.config['training']['optimizer_params']['lr'] = best_params['learning_rate']
        self.config['training']['optimizer_params']['weight_decay'] = best_params['weight_decay']
        self.config['training']['optimizer_params']['betas'] = best_params['betas']
        self.config['training']['optimizer'] = best_params['optimizer']
        self.config['training']['loss_function'] = best_params['loss_function']

        # Train the final model with the best parameters
        self._train_with_configured_hyperparameters()

    def _train_with_configured_hyperparameters(self):
        """
        Train the model using predefined or tuned hyperparameters from the configuration.
        """
        self._update_attributes_from_config()
        self.log_to_mlflow = True  # enable logging of metrics
        patience_counter = 0

        for epoch in range(self.start_epoch + 1, self.num_epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            val_loss, val_accuracy = self.validate(epoch)

            self.logger.log_message(
                f"Epoch {epoch}/{self.num_epochs} Summary: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, "
                f"Val Accuracy = {val_accuracy: .4f}"
            )

            if val_accuracy > self.best_val_accuracy + self.early_stopping_delta:
                self.best_val_accuracy = val_accuracy
                patience_counter = 0
                self.save_checkpoint(epoch, f"{self.checkpoint_dir}/best_model.pth")
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    self.logger.log_message("Early stopping triggered.")
                    break
        self.log_to_mlflow = False  # disable logging of metrics

    def train_one_epoch(self, epoch: int) -> float:
        """
        Train the model for one epoch.

        This function performs a single training epoch, including forward pass, loss computation,
        backpropagation, and optimizer updates. It also logs progress and metrics.

        Args:
            epoch (int): The current epoch number.

        Returns:
            float: The average training loss for the epoch.
        """
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
            correct += torch.as_tensor(predictions == labels, dtype=torch.int).sum().item()
            total += labels.size(0)

            if batch_idx % self.logger.log_interval == 0:
                accuracy = correct / total
                self.logger.log_message(
                    f"Epoch {epoch}, Batch {batch_idx}: Loss = {loss.item():.4f}, Accuracy = {accuracy:.4f}")

        self.scheduler.step()
        train_loss = running_loss / len(self.train_loader)
        train_accuracy = correct / total

        if self.log_to_mlflow:
            self.logger.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_accuracy
            }, step=epoch)

        return train_loss

    def validate(self, epoch: int) -> Tuple[float, float]:
        """
        Validate the model on the validation dataset.

        This function evaluates the model's performance on the validation set, calculating
        metrics such as loss, accuracy, precision, recall, and F1 score. It logs these metrics
        and returns the average validation loss.

        Args:
            epoch (int): The current epoch number.

        Returns:
            float: The average validation loss for the epoch and validation accuracy.
        """
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
        accuracy = torch.as_tensor(all_predictions == np.array(all_labels), dtype=torch.float).mean().item()
        precision = precision_score(all_labels, all_predictions, zero_division=1)
        recall = recall_score(all_labels, all_predictions, zero_division=1)
        f1 = f1_score(all_labels, all_predictions, zero_division=1)

        self.logger.log_message(
            f"Validation: Loss = {val_loss:.4f}, Accuracy = {accuracy:.4f}, Precision = {precision:.4f}, Recall = "
            f"{recall:.4f}, F1 = {f1:.4f}")

        if self.log_to_mlflow:
            self.logger.log_metrics({
                "val_loss": val_loss,
                "val_accuracy": accuracy,
                "val_precision": precision,
                "val_recall": recall,
                "val_f1": f1
            }, step=epoch)

        return val_loss, accuracy

    def train(self):
        """
        Train the model, optionally using Bayesian hyperparameter tuning.
        """
        if self.config['training']['hyperparameter_tuning']['enabled']:
            self.logger.log_message("Hyperparameter tuning is enabled. Starting Bayesian search...")
            self.run_bayesian_search()
        else:
            self.logger.log_message("Hyperparameter tuning is disabled. Using predefined training configuration.")
            self._train_with_configured_hyperparameters()
