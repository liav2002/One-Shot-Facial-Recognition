import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

from utils.logger import Logger
from utils.config_loader import load_config
from data.pairs_dataset import PairsDataset
from config.consts import CONFIG_PATH
from trainer.trainer import Trainer


def test_model(checkpoint_path: str):
    """
    Test the Siamese Network on the test dataset.

    Args:
        config_path (str): Path to the YAML configuration file.
        checkpoint_path (str): Path to the model checkpoint file.
    """
    config = load_config(CONFIG_PATH)
    logger = Logger(config)

    test_df = PairsDataset.load_pairs_from_txt_file(
        config['data']['test_pairs_path'],
        config['data']['lfw_data_path']
    )

    transforms = Compose([
        Resize(config['data']['image_size']),
        ToTensor(),
    ])

    test_loader = DataLoader(
        PairsDataset(test_df, transform=transforms),
        batch_size=config['testing']['batch_size'],
        shuffle=False
    )

    trainer = Trainer(config_path, logger)
    trainer.load_checkpoint(checkpoint_path)

    # Evaluate Model
    logger.start_run(run_name="Siamese_Testing")
    trainer.model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            (img1, img2), labels = batch
            img1, img2, labels = img1.to(trainer.device), img2.to(trainer.device), labels.to(trainer.device)

            outputs = trainer.model(img1, img2).squeeze()
            loss = trainer.loss_fn(outputs, labels.float())
            test_loss += loss.item()

            predictions = (outputs > 0.5).long()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    test_loss /= len(test_loader)
    accuracy = correct / total

    logger.log_message(f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}")
    logger.log_metrics({"test_loss": test_loss, "test_accuracy": accuracy})
    logger.end_run()
