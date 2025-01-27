import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from sklearn.metrics import precision_score, recall_score, f1_score

from src.utils.logger import get_logger
from src.trainer.trainer import Trainer
from src.utils.config_loader import load_config
from src.utils.load_pairs import load_pairs_from_txt_file
from data.pairs_dataset import PairsDataset
from config.consts import CONFIG_PATH, CHECKPOINT_PATH


def test_model(run_name: str) -> None:
    """
    Test the Siamese Network on the test dataset.

    Args:
        run_name (str): The name of the testing run.

    Returns:
        None
    """
    config = load_config(CONFIG_PATH)
    logger = get_logger(config)

    test_df = load_pairs_from_txt_file(
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

    trainer = Trainer(config)
    trainer.load_checkpoint(CHECKPOINT_PATH)

    # Evaluate Model
    logger.start_run(run_name=run_name)
    trainer.model.eval()

    test_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

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

            # Collect all predictions and labels for calculating metrics later
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    test_loss /= len(test_loader)
    accuracy = correct / total
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    # Log metrics
    logger.log_message(f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}")
    logger.log_message(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    logger.log_metrics({
        "test_loss": test_loss,
        "test_accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })
    logger.end_run()
