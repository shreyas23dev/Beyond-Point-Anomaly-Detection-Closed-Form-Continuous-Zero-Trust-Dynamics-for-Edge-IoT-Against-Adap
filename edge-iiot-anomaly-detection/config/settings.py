"""Centralized project configuration."""
import os
from dataclasses import dataclass, field
from pathlib import Path

RANDOM_SEED = 42
SCHEMA_VERSION = "1.0.0"


@dataclass
class PathConfig:
    """Filesystem path configuration for all project directories."""

    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
    )
    data_raw: Path = field(init=False)
    data_processed: Path = field(init=False)
    data_profiling: Path = field(init=False)
    artifacts_models: Path = field(init=False)
    artifacts_metadata: Path = field(init=False)
    artifacts_plots: Path = field(init=False)
    artifacts_results: Path = field(init=False)
    mlruns: Path = field(init=False)

    def __post_init__(self):
        self.data_raw = self.project_root / "data" / "raw"
        self.data_processed = self.project_root / "data" / "processed"
        self.data_profiling = self.project_root / "data" / "profiling"
        self.artifacts_models = self.project_root / "artifacts" / "models"
        self.artifacts_metadata = self.project_root / "artifacts" / "metadata"
        self.artifacts_plots = self.project_root / "artifacts" / "plots"
        self.artifacts_results = self.project_root / "artifacts" / "results"
        self.mlruns = self.project_root / "mlruns"

    def ensure_dirs(self):
        """Create all project directories if they do not exist."""
        for p in [
            self.data_raw,
            self.data_processed,
            self.data_profiling,
            self.artifacts_models,
            self.artifacts_metadata,
            self.artifacts_plots,
            self.artifacts_results,
            self.mlruns,
        ]:
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class DatasetConfig:
    """Configuration for the Edge-IIoTset dataset."""

    filename: str = "DNN-EdgeIIoT-dataset.csv"
    kaggle_dataset: str = (
        "mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot"
    )
    kaggle_file: str = (
        "Edge-IIoTset dataset/Selected dataset for ML and DL/"
        "DNN-EdgeIIoT-dataset.csv"
    )
    target_column: str = "Attack_label"
    attack_type_column: str = "Attack_type"
    normal_label: int = 0
    attack_label: int = 1
    columns_to_drop: list[str] = field(
        default_factory=lambda: [
            "frame.time",
            "ip.src_host",
            "ip.dst_host",
            "arp.src.proto_ipv4",
            "arp.dst.proto_ipv4",
            "arp.src.hw_mac",
            "arp.dst.hw_mac",
            "tcp.payload",
            "tcp.options",
            "mqtt.msg",
            "http.file_data",
            "http.request.full_uri",
            "icmp.unused",
            "dns.qry.name",
            "http.request.uri.query",
            "tcp.segment_data",
            "http.tls_port",
        ]
    )


@dataclass
class SplitConfig:
    """Train / validation / test split ratios."""

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_state: int = RANDOM_SEED


@dataclass
class ModelConfig:
    """Isolation Forest model hyperparameters and tuning ranges."""

    # Default hyperparameters
    n_estimators: int = 200
    max_samples: str | float = "auto"
    max_features: float = 1.0
    bootstrap: bool = False
    random_state: int = RANDOM_SEED
    n_jobs: int = 8

    # Tuning ranges
    n_estimators_range: tuple[int, int] = (100, 500)
    max_samples_choices: list = field(
        default_factory=lambda: ["auto", 0.5, 0.7, 1.0]
    )
    max_features_range: tuple[float, float] = (0.5, 1.0)
    bootstrap_choices: list[bool] = field(default_factory=lambda: [True, False])

    # Tuning settings
    n_optuna_trials: int = 10
    cv_folds: int = 3

    # Threshold
    threshold_methods: list[str] = field(
        default_factory=lambda: [
            "percentile",
            "f1_optimized",
            "pr_curve_inflection",
            "iqr_based",
            "contamination_sweep",
        ]
    )
    threshold_tie_epsilon: float = 0.001


@dataclass
class APIConfig:
    """FastAPI serving configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    max_batch_size: int = 1000
    title: str = "Edge-IIoT Anomaly Detection API"
    version: str = SCHEMA_VERSION


@dataclass
class MLflowConfig:
    """MLflow experiment tracking configuration."""

    experiment_name: str = "edge-iiot-anomaly-detection"
    tracking_uri: str = field(
    	default_factory=lambda: (
            Path(__file__).resolve().parent.parent / "mlruns"
    	).resolve().as_uri()
    )


@dataclass
class ProjectConfig:
    """Top-level project configuration aggregating all sub-configs."""

    paths: PathConfig = field(default_factory=PathConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    api: APIConfig = field(default_factory=APIConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    random_seed: int = RANDOM_SEED
    schema_version: str = SCHEMA_VERSION


# Global instance
config = ProjectConfig()
