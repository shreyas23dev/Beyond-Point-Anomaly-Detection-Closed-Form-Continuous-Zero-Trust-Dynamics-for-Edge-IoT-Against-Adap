"""Data loading utilities."""
import logging
from pathlib import Path
from typing import Generator

import pandas as pd

from config.settings import config

logger = logging.getLogger(__name__)

class DataLoader:
    """Utility class for loading the Edge-IIoTset dataset."""
    
    def __init__(self, filepath: Path, low_memory: bool = False):
        """Initialize the DataLoader.
        
        Args:
            filepath: Path to the dataset CSV file.
            low_memory: Whether to use pandas low_memory mode (default False to avoid mixed types).
        """
        self.filepath = filepath
        self.low_memory = low_memory
        
        if not self.filepath.exists():
            raise FileNotFoundError(f"Dataset not found at {self.filepath}")

    def load(self) -> pd.DataFrame:
        """Load the entire dataset into memory.
        
        Returns:
            pd.DataFrame containing the dataset.
        """
        logger.info(f"Loading dataset from {self.filepath}...")
        df = pd.read_csv(self.filepath, low_memory=self.low_memory)
        
        memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns.")
        logger.info(f"Memory usage: {memory_usage:.2f} MB")
        
        if not self.validate(df):
            logger.warning("Dataset validation failed. Missing required columns.")
            
        return df
        
    def load_chunked(self, chunksize: int = 100_000) -> Generator[pd.DataFrame, None, None]:
        """Load the dataset in chunks for memory-constrained environments.
        
        Args:
            chunksize: Number of rows per chunk.
            
        Yields:
            pd.DataFrame chunks.
        """
        logger.info(f"Loading dataset from {self.filepath} in chunks of {chunksize}...")
        for chunk_idx, chunk in enumerate(pd.read_csv(self.filepath, low_memory=self.low_memory, chunksize=chunksize)):
            logger.info(f"Loaded chunk {chunk_idx + 1} ({len(chunk)} rows)")
            yield chunk

    def validate(self, df: pd.DataFrame) -> bool:
        """Validate that the dataset contains required columns.
        
        Args:
            df: DataFrame to validate.
            
        Returns:
            True if valid, False otherwise.
        """
        required_columns = [config.dataset.target_column, config.dataset.attack_type_column]
        missing = [col for col in required_columns if col not in df.columns]
        
        if missing:
            logger.error(f"Dataset is missing required columns: {missing}")
            return False
            
        return True
