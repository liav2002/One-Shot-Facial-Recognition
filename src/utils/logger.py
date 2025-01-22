import os
import logging
import mlflow
from datetime import datetime

_logger_instance = None  # Global logger instance


class Logger:
    def __init__(self, config: dict):
        self.logs_dir = config['logging']['logs_dir']
        self.log_interval = config['logging']['log_interval']
        self.mlflow_experiment = f"{config['logging']['mlflow_experiment']}"

        os.makedirs(self.logs_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(self.logs_dir, f"log_{timestamp}.log")

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file),
            ]
        )
        self.logger = logging.getLogger()

        mlflow.set_experiment(self.mlflow_experiment)
        self.mlflow_run = None

    def start_run(self, run_name: str = None):
        self.mlflow_run = mlflow.start_run(run_name=run_name)

    def log_metrics(self, metrics: dict, step: int = None):
        if self.mlflow_run:
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)

    def log_params(self, params: dict):
        if self.mlflow_run:
            mlflow.log_params(params)

    def log_artifacts(self, artifact_dir_path: str):
        if self.mlflow_run:
            mlflow.log_artifacts(artifact_dir_path)

    def log_message(self, message: str, level: str = "info"):
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(message)

    def end_run(self):
        if self.mlflow_run:
            mlflow.end_run()
            self.mlflow_run = None


def get_logger(config: dict = None) -> Logger:
    """
    Retrieve the global Logger instance. If it does not exist, create it.

    Args:
        config (dict): Configuration dictionary. Required on first initialization.

    Returns:
        Logger: The global Logger instance.
    """
    global _logger_instance
    if _logger_instance is None:
        if config is None:
            raise ValueError("Logger must be initialized with a config before accessing.")
        _logger_instance = Logger(config)
    return _logger_instance
