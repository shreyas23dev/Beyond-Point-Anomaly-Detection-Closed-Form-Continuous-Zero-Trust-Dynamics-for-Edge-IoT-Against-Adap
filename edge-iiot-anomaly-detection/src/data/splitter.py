"""Data splitting utilities."""
import logging
from typing import NamedTuple

import pandas as pd
from sklearn.model_selection import train_test_split

from config.settings import config, RANDOM_SEED

logger = logging.getLogger(__name__)

class SplitResult(NamedTuple):
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_val: pd.Series
    y_test: pd.Series
    attack_types_val: pd.Series
    attack_types_test: pd.Series

class DataSplitter:
    """Utility class for splitting the Edge-IIoTset dataset."""
    
    def __init__(
        self, 
        train_ratio: float = config.split.train_ratio, 
        val_ratio: float = config.split.val_ratio, 
        test_ratio: float = config.split.test_ratio,
        random_state: int = config.split.random_state
    ):
        """Initialize the DataSplitter.
        
        Args:
            train_ratio: Proportion of ALL data to use for training.
            val_ratio: Proportion of ALL data to use for validation.
            test_ratio: Proportion of ALL data to use for testing.
            random_state: Random seed for reproducibility.
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state

    def split(self, df: pd.DataFrame, target_col: str = config.dataset.target_column, attack_type_col: str = config.dataset.attack_type_column) -> SplitResult:
        """Split the dataset into train, validation, and test sets.
        
        Training set contains ONLY normal traffic.
        Validation and test sets contain a stratified mix of normal + attack traffic.
        
        Args:
            df: The full dataset.
            target_col: Name of the binary target column.
            attack_type_col: Name of the detailed attack type column.
            
        Returns:
            SplitResult containing all data splits.
        """
        logger.info("Splitting dataset...")
        
        # 1. Separate normal and attack traffic
        normal_mask = df[target_col] == config.dataset.normal_label
        df_normal = df[normal_mask].copy()
        df_attack = df[~normal_mask].copy()
        
        total_rows = len(df)
        n_train_target = int(total_rows * self.train_ratio)
        n_normal = len(df_normal)
        
        if n_normal < n_train_target:
            logger.warning(f"Not enough normal samples ({n_normal}) to meet training ratio target ({n_train_target}). Using all normal samples for training except 10% for val/test.")
            # If we don't have enough normal data, just use 90% of it for training
            n_train_actual = int(n_normal * 0.9)
        else:
            n_train_actual = n_train_target
            
        logger.info(f"Total rows: {total_rows}. Target train size: {n_train_target}. Actual train size: {n_train_actual}")
        
        # 2. Split normal data: train vs (val + test)
        # We don't stratify normal data by attack type because it's all "Normal"
        train_normal, val_test_normal = train_test_split(
            df_normal, 
            train_size=n_train_actual, 
            random_state=self.random_state
        )
        
        # 3. Split remaining normal data between val and test
        # Determine proportion based on val_ratio / (val_ratio + test_ratio)
        val_prop = self.val_ratio / (self.val_ratio + self.test_ratio)
        val_normal, test_normal = train_test_split(
            val_test_normal,
            train_size=val_prop,
            random_state=self.random_state
        )
        
        # 4. Split attack data between val and test (stratified by attack type)
        val_attack, test_attack = train_test_split(
            df_attack,
            train_size=val_prop,
            stratify=df_attack[attack_type_col],
            random_state=self.random_state
        )
        
        # 5. Combine normal and attack for val and test
        val_combined = pd.concat([val_normal, val_attack]).sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        test_combined = pd.concat([test_normal, test_attack]).sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        # 6. Extract features and labels
        # X_train gets NO label columns
        X_train = train_normal.drop(columns=[target_col, attack_type_col])
        
        X_val = val_combined.drop(columns=[target_col, attack_type_col])
        y_val = val_combined[target_col]
        attack_types_val = val_combined[attack_type_col]
        
        X_test = test_combined.drop(columns=[target_col, attack_type_col])
        y_test = test_combined[target_col]
        attack_types_test = test_combined[attack_type_col]
        
        result = SplitResult(
            X_train=X_train.reset_index(drop=True),
            X_val=X_val,
            X_test=X_test,
            y_val=y_val,
            y_test=y_test,
            attack_types_val=attack_types_val,
            attack_types_test=attack_types_test
        )
        
        self.validate_split(result, total_rows)
        return result
        
    def validate_split(self, result: SplitResult, total_rows: int) -> bool:
        """Validate the split result for correctness."""
        logger.info("Validating splits...")
        
        n_train = len(result.X_train)
        n_val = len(result.X_val)
        n_test = len(result.X_test)
        
        # Check total rows
        if n_train + n_val + n_test != total_rows:
            logger.warning(f"Row count mismatch! Total: {total_rows}, Sum of splits: {n_train + n_val + n_test}")
            
        logger.info(f"Train size: {n_train} ({n_train/total_rows*100:.1f}%) - ALL NORMAL")
        logger.info(f"Val size: {n_val} ({n_val/total_rows*100:.1f}%) - Normal: {sum(result.y_val==config.dataset.normal_label)}, Attack: {sum(result.y_val==config.dataset.attack_label)}")
        logger.info(f"Test size: {n_test} ({n_test/total_rows*100:.1f}%) - Normal: {sum(result.y_test==config.dataset.normal_label)}, Attack: {sum(result.y_test==config.dataset.attack_label)}")
        
        # Ensure no target columns in X
        assert config.dataset.target_column not in result.X_train.columns
        assert config.dataset.attack_type_column not in result.X_train.columns
        
        return True
