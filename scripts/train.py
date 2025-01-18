from trainer.trainer import Trainer
from utils.logger import Logger
from utils.config_loader import load_config
from config.consts import CONFIG_PATH

def train_model():
    """
    Train the Siamese Network using the Trainer class.
    """
    config = load_config(CONFIG_PATH)
    logger = Logger(config)
    trainer = Trainer(config, logger)

    logger.start_run(run_name="Siamese_Training")
    trainer.train()
    logger.end_run()
