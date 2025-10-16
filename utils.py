# utils.py - Utility functions for the ML pipeline
import os
import logging
from datetime import datetime


def setup_logging():
    """Set up logging configuration."""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/ml_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


def create_experiment_dir():
    """Create a directory for the current experiment."""
    # Create experiments directory if it doesn't exist
    os.makedirs('experiments', exist_ok=True)

    # Create a timestamped directory for this experiment
    experiment_dir = os.path.join('experiments', f'experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(experiment_dir, exist_ok=True)

    return experiment_dir


def get_polynomial_feature_name(degree, feature_names):
    """Generate feature names for polynomial features."""
    import itertools

    feature_names = list(feature_names)
    n_features = len(feature_names)

    if degree == 0:
        return ['1']

    feature_names_list = []

    for d in range(1, degree + 1):
        for combi in itertools.combinations_with_replacement(list(range(n_features)), d):
            name = ''
            counts = {}

            for index in combi:
                if index in counts:
                    counts[index] += 1
                else:
                    counts[index] = 1

            for index, count in counts.items():
                if count == 1:
                    name += feature_names[index] + ' '
                else:
                    name += feature_names[index] + '^' + str(count) + ' '

            feature_names_list.append(name.strip())

    return feature_names_list


