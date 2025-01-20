from trainer.trainer import Trainer
from utils.logger import get_logger
from utils.config_loader import load_config
from config.consts import CONFIG_PATH

def train_model(run_name = None):
    """
    Train the Siamese Network using the Trainer class.
    """
    config = load_config(CONFIG_PATH)
    logger = get_logger(config)
    trainer = Trainer(config)

    # logger.start_run(run_name=run_name)
    # trainer.train()
    # logger.end_run()
