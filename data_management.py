# data_management.py - Module for data loading, validation, and splitting
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Class for loading data from various sources."""

    def load_excel(self, file_path):
        """Load data from Excel file."""
        try:
            data = pd.read_excel(file_path)
            logger.info(f"Successfully loaded data with shape: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def load_csv(self, file_path):
        """Load data from CSV file."""
        try:
            data = pd.read_csv(file_path)
            logger.info(f"Successfully loaded data with shape: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise


class DataValidator:
    """Class for validating input data."""

    def validate(self, data):
        """Validate input data for quality issues."""
        results = {}

        # Check for missing values
        missing_data = data.isnull().sum()
        results['missing_values'] = missing_data.to_dict()

        # Check for infinite values
        inf_count = np.isinf(data.select_dtypes(include=['float64', 'int64'])).sum()
        results['infinite_values'] = inf_count.to_dict()

        # Check for duplicates
        results['duplicate_rows'] = data.duplicated().sum()

        # Check for outliers using IQR (only on numeric columns)
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        outlier_counts = {}

        for col in numeric_cols:
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_counts[col] = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()

        results['outliers'] = outlier_counts

        return results


class DataSplitter:
    """Class for splitting data into train/test sets."""

    def __init__(self, random_state=42):
        self.random_state = random_state

    def train_test_split(self, X, y, test_size=0.2, stratify_by=None):
        """Split data into train and test sets."""
        # If stratify_by is provided, use it for stratified splitting
        stratify = None
        if stratify_by is not None:
            if isinstance(stratify_by, str) and stratify_by in X.columns:
                stratify = X[stratify_by]
            elif isinstance(stratify_by, pd.Series) and len(stratify_by) == len(X):
                stratify = stratify_by

        # For regression problems, we can create bins for stratification
        if stratify is None and len(y.shape) == 1:
            # Create 5 bins for the target variable
            y_binned = pd.qcut(y, q=5, duplicates='drop', labels=False)
            stratify = y_binned

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state,
            stratify=stratify
        )

        logger.info(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test