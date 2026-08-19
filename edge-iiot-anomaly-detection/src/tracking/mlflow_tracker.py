"""MLflow tracking integration."""

import logging
from pathlib import Path

import mlflow

from config.settings import config

logger = logging.getLogger(__name__)


class MLflowTracker:
    """Wrapper for MLflow experiment tracking."""

    def __init__(
        self,
        experiment_name: str = config.mlflow.experiment_name,
        tracking_uri: str = config.mlflow.tracking_uri,
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri

        # Ensure the local MLflow tracking directory exists
        mlruns_path = config.paths.mlruns.resolve()
        mlruns_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Setting MLflow tracking URI: {self.tracking_uri}")
        mlflow.set_tracking_uri(self.tracking_uri)

        logger.info(f"Setting MLflow experiment: {self.experiment_name}")

        experiment = mlflow.get_experiment_by_name(self.experiment_name)

        if experiment is None:
            artifact_uri = mlruns_path.as_uri()

            logger.info(
                f"Creating MLflow experiment '{self.experiment_name}' "
                f"with artifact location: {artifact_uri}"
            )

            experiment_id = mlflow.create_experiment(
                name=self.experiment_name,
                artifact_location=artifact_uri,
            )

            logger.info(f"Created experiment ID: {experiment_id}")

        mlflow.set_experiment(self.experiment_name)

    def start_run(self, run_name: str | None = None):
        """Start an MLflow run."""
        return mlflow.start_run(run_name=run_name)

    def log_params(self, params: dict):
        """Log parameters to the current run."""
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict, step: int | None = None):
        """Log metrics to the current run."""
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: Path | str, artifact_path: str | None = None):
        """Log a local file as an artifact."""
        mlflow.log_artifact(str(local_path), artifact_path)

    def log_artifacts(self, local_dir: Path | str, artifact_path: str | None = None):
        """Log all files in a local directory as artifacts."""
        mlflow.log_artifacts(str(local_dir), artifact_path)

    def log_dict_as_json(self, data: dict, artifact_file: str):
        """Log a dictionary directly as a JSON artifact."""
        mlflow.log_dict(data, artifact_file)

    def end_run(self):
        """End the current run."""
        mlflow.end_run()