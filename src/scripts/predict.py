import torch
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
from src.utils.logger import get_logger
from src.utils.config_loader import load_config
from src.model.siamese_network import SiameseNetwork  # Adjust import if needed
from config.consts import CONFIG_PATH, CHECKPOINT_PATH


def load_model(config, checkpoint_path):
    """
    Load the trained Siamese Network model.

    Args:
        config (dict): Configuration settings.
        checkpoint_path (str): Path to the model checkpoint.

    Returns:
        model (torch.nn.Module): Loaded model in evaluation mode.
        device (torch.device): The device (CPU/GPU) where the model is loaded.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model (ensure it matches training architecture)
    model = SiameseNetwork(config['model'], init_weights=False)

    # Load checkpoint and extract only the model weights
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Ensure only model weights are loaded
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    return model, device


def preprocess_image(image_path, transform):
    """
    Preprocesses an image for model input.

    Args:
        image_path (str): Path to the image file.
        transform (Compose): Transformations to apply.

    Returns:
        torch.Tensor: Transformed image.
    """
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    return transform(image).unsqueeze(0)  # Add batch dimension


def predict(run_name: str, person1: str, image1_path: str, person2: str, image2_path: str) -> None:
    """
    Predicts whether two images belong to the same person.

    Args:
        run_name (str): Name of the prediction run.
        person1 (str): Name of the first person.
        image1_path (str): Path to the first person's image.
        person2 (str): Name of the second person.
        image2_path (str): Path to the second person's image.

    Returns:
        None
    """
    # Load config and initialize logger
    config = load_config(CONFIG_PATH)
    logger = get_logger(config)

    # Define image transformations
    transforms = Compose([
        Resize(config['data']['image_size']),
        ToTensor(),
    ])

    # Load the model
    model, device = load_model(config, CHECKPOINT_PATH)

    # Preprocess images
    image1 = preprocess_image(image1_path, transforms).to(device)
    image2 = preprocess_image(image2_path, transforms).to(device)

    # Predict similarity
    with torch.no_grad():
        output = model(image1, image2).squeeze().item()

    # Use threshold from config
    threshold = config['testing']['threshold']
    is_same_person = output > threshold
    prediction_text = f"{person1} and {person2} {'ARE' if is_same_person else 'ARE NOT'} the same person."

    # Log results
    logger.start_run(run_name=run_name)
    logger.log_message(f"Prediction result: {prediction_text}")
    logger.log_message(f"Similarity score: {output:.4f}")
    logger.end_run()

    # Print result
    print(prediction_text)
