import os
import logging
import mlflow
from datetime import datetime


class Logger:
    def __init__(self, config: dict):
        """
        Logger to handle console, file, and MLFlow logging.

        Args:
            config (dict): Configuration dictionary containing logging parameters.
        """
        self.logs_dir = config['logging']['logs_dir']
        self.mlflow_experiment = config['logging']['mlflow_experiment']
        self.log_interval = config['logging']['log_interval']

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
        """
        Start an MLFlow run.

        Args:
            run_name (str, optional): Name of the MLFlow run. Defaults to None.
        """
        self.mlflow_run = mlflow.start_run(run_name=run_name)

    def log_metrics(self, metrics: dict, step: int = None):
        """
        Log metrics to MLFlow.

        Args:
            metrics (dict): Dictionary of metric names and values.
            step (int, optional): Step or epoch for the metrics. Defaults to None.
        """
        if self.mlflow_run:
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)

    def log_params(self, params: dict):
        """
        Log parameters to MLFlow.

        Args:
            params (dict): Dictionary of parameter names and values.
        """
        if self.mlflow_run:
            mlflow.log_params(params)

    def log_artifacts(self, artifact_path: str):
        """
        Log artifacts (e.g., model checkpoints) to MLFlow.

        Args:
            artifact_path (str): Path to the artifact directory or file.
        """
        if self.mlflow_run:
            mlflow.log_artifacts(artifact_path)

    def log_message(self, message: str, level: str = "info"):
        """
        Log a message to console and file.

        Args:
            message (str): Message to log.
            level (str): Logging level (e.g., "info", "warning", "error"). Defaults to "info".
        """
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(message)

    def end_run(self):
        """
        End the current MLFlow run.
        """
        if self.mlflow_run:
            mlflow.end_run()
            self.mlflow_run = None
