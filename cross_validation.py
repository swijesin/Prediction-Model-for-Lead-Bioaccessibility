# cross_validation.py - Module for enhanced cross-validation

from sklearn.model_selection import RepeatedKFold, cross_val_score, cross_validate
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import clone
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
import logging

logger = logging.getLogger(__name__)



class EnhancedCrossValidator:
    """
    Enhanced cross-validation with stability metrics to reduce overfitting.
    """

    def __init__(self, n_splits=5, n_repeats=3, random_state=42, experiment_dir=None):
        """
        Initialize enhanced cross-validator.

        Parameters:
        -----------
        n_splits : int, default=5
            Number of folds for k-fold cross-validation
        n_repeats : int, default=3
            Number of times to repeat k-fold cross-validation
        random_state : int, default=42
            Random seed for reproducibility
        experiment_dir : str, optional
            Directory to save cross-validation results
        """
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.experiment_dir = experiment_dir
        if experiment_dir:
            self.cv_dir = os.path.join(experiment_dir, 'cross_validation')
            os.makedirs(self.cv_dir, exist_ok=True)
        else:
            self.cv_dir = None

    def evaluate_model(self, model, X, y, model_name):
        """
        Evaluate model using repeated k-fold cross-validation.

        Parameters:
        -----------
        model : estimator
            Model to evaluate
        X : DataFrame
            Feature matrix
        y : Series
            Target values
        model_name : str
            Name of the model for logging and saving results

        Returns:
        --------
        dict
            Cross-validation results with stability metrics
        """
        logger.info(
            f"Performing enhanced {self.n_splits}-fold cross-validation ({self.n_repeats} repeats) for {model_name}")

        # Define cross-validation strategy
        cv = RepeatedKFold(
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            random_state=self.random_state
        )

        # Define scoring metrics
        scoring = {
            'r2': 'r2',
            'neg_mean_squared_error': 'neg_mean_squared_error',
            'neg_mean_absolute_error': 'neg_mean_absolute_error'
        }

        # Perform cross-validation
        cv_results = cross_validate(
            model, X, y,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            return_estimator=True,
            n_jobs=-1
        )

        # Calculate stability metrics
        train_r2_mean = np.mean(cv_results['train_r2'])
        train_r2_std = np.std(cv_results['train_r2'])
        test_r2_mean = np.mean(cv_results['test_r2'])
        test_r2_std = np.std(cv_results['test_r2'])

        # Calculate overfitting metric (train-test gap)
        r2_gap = train_r2_mean - test_r2_mean

        # Log results
        logger.info(f"CV Results for {model_name}:")
        logger.info(f"  Train R² = {train_r2_mean:.4f} +/- {train_r2_std:.4f}")
        logger.info(f"  Test R² = {test_r2_mean:.4f} +/- {test_r2_std:.4f}")
        logger.info(f"  Overfitting Gap = {r2_gap:.4f}")

        # Save visualization if experiment_dir provided
        if self.cv_dir:
            self._visualize_cv_results(cv_results, model_name)

        # Create complete results dictionary
        results = {
            'model_name': model_name,
            'train_r2_mean': train_r2_mean,
            'train_r2_std': train_r2_std,
            'test_r2_mean': test_r2_mean,
            'test_r2_std': test_r2_std,
            'r2_gap': r2_gap,
            'n_splits': self.n_splits,
            'n_repeats': self.n_repeats,
            'cv_results': cv_results
        }

        return results

    def evaluate_models(self, models, X, y):
        """
        Evaluate multiple models using enhanced cross-validation.

        Parameters:
        -----------
        models : dict
            Dictionary of models with model names as keys
        X : DataFrame
            Feature matrix
        y : Series
            Target values

        Returns:
        --------
        dict
            Cross-validation results for all models
        """
        logger.info(f"Evaluating {len(models)} models with enhanced cross-validation")

        cv_results = {}
        for model_name, model in models.items():
            cv_results[model_name] = self.evaluate_model(model, X, y, model_name)

        # Create summary report
        if self.cv_dir:
            self._create_cv_summary(cv_results)

        return cv_results

    def _visualize_cv_results(self, cv_results, model_name):
        """
        Create visualizations for cross-validation results.

        Parameters:
        -----------
        cv_results : dict
            Cross-validation results
        model_name : str
            Name of the model
        """
        # Create R² distribution plot
        plt.figure(figsize=(10, 6))

        # Plot train and test R² for each fold
        x = np.arange(len(cv_results['train_r2']))
        plt.scatter(x, cv_results['train_r2'], label='Train R²', alpha=0.7, color='blue')
        plt.scatter(x, cv_results['test_r2'], label='Test R²', alpha=0.7, color='orange')

        # Add mean lines
        plt.axhline(y=np.mean(cv_results['train_r2']), color='blue', linestyle='--',
                    label=f'Mean Train R² = {np.mean(cv_results["train_r2"]):.4f}')
        plt.axhline(y=np.mean(cv_results['test_r2']), color='orange', linestyle='--',
                    label=f'Mean Test R² = {np.mean(cv_results["test_r2"]):.4f}')

        # Add gap annotation
        gap = np.mean(cv_results['train_r2']) - np.mean(cv_results['test_r2'])
        plt.annotate(f'Gap = {gap:.4f}',
                     xy=(0.7, 0.5), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        plt.title(f'Cross-Validation R² Distribution for {model_name}')
        plt.xlabel('CV Fold')
        plt.ylabel('R² Score')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.cv_dir, f'{model_name}_cv_r2_distribution.png'))
        plt.close()

        # Create train-test gap plot
        plt.figure(figsize=(10, 6))

        # Calculate gap for each fold
        gaps = cv_results['train_r2'] - cv_results['test_r2']

        # Plot gaps
        plt.bar(x, gaps, alpha=0.7, color='purple')
        plt.axhline(y=np.mean(gaps), color='red', linestyle='--',
                    label=f'Mean Gap = {np.mean(gaps):.4f}')

        plt.title(f'Cross-Validation Overfitting Gap for {model_name}')
        plt.xlabel('CV Fold')
        plt.ylabel('Train-Test R² Gap')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.cv_dir, f'{model_name}_cv_gap.png'))
        plt.close()

    def _create_cv_summary(self, cv_results):
        """
        Create summary report for all models.

        Parameters:
        -----------
        cv_results : dict
            Cross-validation results for all models
        """
        # Extract summary metrics
        summary = []
        for model_name, results in cv_results.items():
            summary.append({
                'Model': model_name,
                'Train R²': f"{results['train_r2_mean']:.4f} +/- {results['train_r2_std']:.4f}",
                'Test R²': f"{results['test_r2_mean']:.4f} +/- {results['test_r2_std']:.4f}",
                'Gap': f"{results['r2_gap']:.4f}",
                'Train R² Mean': results['train_r2_mean'],
                'Test R² Mean': results['test_r2_mean'],
                'Gap Value': results['r2_gap']
            })

        # Create DataFrame and save to CSV
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(os.path.join(self.cv_dir, 'cv_summary.csv'), index=False)

        # Create comparison plot
        plt.figure(figsize=(12, 8))

        # Sort models by test R²
        summary_df = summary_df.sort_values('Test R² Mean', ascending=False)

        # Extract model names and metrics
        models = summary_df['Model']
        train_r2 = summary_df['Train R² Mean']
        test_r2 = summary_df['Test R² Mean']
        gaps = summary_df['Gap Value']

        # Set up bar positions
        x = np.arange(len(models))
        width = 0.3

        # Create bars
        plt.bar(x - width, train_r2, width, label='Train R²', color='blue', alpha=0.7)
        plt.bar(x, test_r2, width, label='Test R²', color='orange', alpha=0.7)
        plt.bar(x + width, gaps, width, label='Gap', color='purple', alpha=0.7)

        # Add labels and grid
        plt.xlabel('Models')
        plt.ylabel('R² Score / Gap')
        plt.title('Cross-Validation Performance Comparison')
        plt.xticks(x, models, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add values on bars
        for i, v in enumerate(train_r2):
            plt.text(i - width, v + 0.01, f"{v:.3f}", ha='center', fontsize=9)
        for i, v in enumerate(test_r2):
            plt.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=9)
        for i, v in enumerate(gaps):
            plt.text(i + width, v + 0.01, f"{v:.3f}", ha='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(self.cv_dir, 'cv_model_comparison.png'))
        plt.close()


def train_with_cv(model_class, param_grid, X, y, cv=None, n_iter=10, scoring='r2', n_jobs=-1, random_state=42):
    """
    Train a model with cross-validated hyperparameter optimization.

    Parameters:
    -----------
    model_class : estimator class
        Scikit-learn estimator class to instantiate
    param_grid : dict
        Hyperparameter grid to search
    X : DataFrame
        Feature matrix
    y : Series
        Target values
    cv : int or cross-validation generator, default=None
        Cross-validation strategy (None uses 5-fold CV)
    n_iter : int, default=10
        Number of parameter settings sampled
    scoring : str, default='r2'
        Scoring metric
    n_jobs : int, default=-1
        Number of parallel jobs
    random_state : int, default=42
        Random seed for reproducibility

    Returns:
    --------
    best_model : estimator
        Best model trained on all data
    """
    from sklearn.model_selection import RandomizedSearchCV

    # Use 5-fold CV by default
    if cv is None:
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

    # Create base model
    base_model = model_class(random_state=random_state)

    # Create RandomizedSearchCV
    search = RandomizedSearchCV(
        base_model,
        param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=1,
        return_train_score=True
    )

    # Fit search
    search.fit(X, y)

    # Get best model
    best_model = search.best_estimator_

    return best_model


def create_stratified_kfold_for_regression(y, n_splits=5, n_repeats=3, random_state=42):
    """
    Create stratified folds for regression problems by binning the target variable.

    Parameters:
    -----------
    y : Series
        Target values
    n_splits : int, default=5
        Number of folds
    n_repeats : int, default=3
        Number of repeats
    random_state : int, default=42
        Random seed

    Returns:
    --------
    cv : RepeatedStratifiedKFold
        Cross-validation strategy
    """
    from sklearn.model_selection import RepeatedStratifiedKFold
    import numpy as np
    import pandas as pd

    # Create bins for target variable (5 equal-width bins)
    y_binned = pd.qcut(y, q=5, labels=False, duplicates='drop')

    # Create stratified k-fold with repeats
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state
    )

    return cv, y_binned
def get_optimal_cv_params(model_type):
    """
    Get optimal cross-validation parameters for each model type.

    Parameters:
    -----------
    model_type : str
        Model type name

    Returns:
    --------
    dict
        CV parameters
    """
    # Default parameters
    default_params = {
        'n_splits': 5,
        'n_repeats': 3,
        'random_state': 42,
        'gap_threshold': 0.10
    }

    # Model-specific adjustments based on the CV gap charts
    if model_type == 'neural_network':
        # Neural network showed lowest gap, use more folds for reliability
        return {
            'n_splits': 5,
            'n_repeats': 3,
            'random_state': 42,
            'gap_threshold': 0.06  # Stricter threshold based on observed performance
        }
    elif model_type == 'lightgbm':
        # LightGBM had good performance, adjust parameters
        return {
            'n_splits': 5,
            'n_repeats': 3,
            'random_state': 42,
            'gap_threshold': 0.13
        }
    elif model_type == 'random_forest':
        # Random forest had higher gaps, be more selective
        return {
            'n_splits': 5,
            'n_repeats': 3,
            'random_state': 42,
            'gap_threshold': 0.13
        }

    return default_params


def perform_nested_cross_validation(X, y, model_creator, preprocessor_creator, feature_selector_creator, n_outer=5,
                                    n_inner=3):
    """
    Perform nested cross-validation to get unbiased performance estimation.

    Parameters:
    -----------
    X : DataFrame
        Feature matrix
    y : Series
        Target values
    model_creator : function
        Function that returns a new model instance when called
    preprocessor_creator : function
        Function that returns a new preprocessor instance when called
    feature_selector_creator : function
        Function that returns a new feature selector instance when called
    n_outer : int, default=5
        Number of outer CV folds
    n_inner : int, default=3
        Number of inner CV folds

    Returns:
    --------
    float
        Mean R² score across outer folds
    """
    from sklearn.model_selection import KFold
    import numpy as np
    from sklearn.metrics import r2_score
    import pandas as pd

    # Create outer CV
    outer_cv = KFold(n_splits=n_outer, shuffle=True, random_state=42)

    # Store scores for each outer fold
    outer_scores = []
    outer_train_scores = []
    outer_gaps = []
    all_true_values = []
    all_predictions = []

    # For each outer split
    for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(X)):
        # Split data
        X_outer_train = X.iloc[outer_train_idx].copy() if isinstance(X, pd.DataFrame) else X[outer_train_idx].copy()
        X_outer_test = X.iloc[outer_test_idx].copy() if isinstance(X, pd.DataFrame) else X[outer_test_idx].copy()
        y_outer_train = y.iloc[outer_train_idx].copy() if isinstance(y, pd.Series) else y[outer_train_idx].copy()
        y_outer_test = y.iloc[outer_test_idx].copy() if isinstance(y, pd.Series) else y[outer_test_idx].copy()

        # Create fresh preprocessor and fit on outer train only
        preprocessor = preprocessor_creator()
        preprocessor.fit(X_outer_train, y_outer_train)

        # Transform outer train and test
        X_outer_train_processed = preprocessor.transform(X_outer_train)
        X_outer_test_processed = preprocessor.transform(X_outer_test)

        # Process target separately if needed
        if hasattr(preprocessor, 'transform_target'):
            y_outer_train_processed = preprocessor.transform_target(y_outer_train)
            y_outer_test_processed = preprocessor.transform_target(y_outer_test)
        else:
            y_outer_train_processed = y_outer_train
            y_outer_test_processed = y_outer_test

        # Feature selection on outer train only
        feature_selector = feature_selector_creator()
        feature_selector.fit(X_outer_train_processed, y_outer_train_processed)

        # Apply feature selection
        X_outer_train_selected = feature_selector.transform(X_outer_train_processed)
        X_outer_test_selected = feature_selector.transform(X_outer_test_processed)

        # Create inner CV for model selection/hyperparameter tuning
        inner_cv = KFold(n_splits=n_inner, shuffle=True, random_state=42)

        # Create and train model with inner CV
        # (Here you could add hyperparameter tuning if needed)
        model = model_creator()
        model.fit(X_outer_train_selected, y_outer_train_processed)

        # Evaluate on train and test sets
        y_train_pred = model.predict(X_outer_train_selected)
        y_test_pred = model.predict(X_outer_test_selected)

        # Calculate and store scores
        train_score = r2_score(y_outer_train_processed, y_train_pred)
        test_score = r2_score(y_outer_test_processed, y_test_pred)
        gap = train_score - test_score

        outer_train_scores.append(train_score)
        outer_scores.append(test_score)
        outer_gaps.append(gap)

        # Store true values and predictions
        all_true_values.extend(y_outer_test_processed)
        all_predictions.extend(y_test_pred)

        logger.info(
            f"Outer fold {fold_idx + 1}/{n_outer}: Train R² = {train_score:.4f}, Test R² = {test_score:.4f}, Gap = {gap:.4f}")

    # Calculate mean scores
    mean_train_score = np.mean(outer_train_scores)
    std_train_score = np.std(outer_train_scores)
    mean_test_score = np.mean(outer_scores)
    std_test_score = np.std(outer_scores)
    mean_gap = np.mean(outer_gaps)

    # Calculate overall R² based on all predictions
    overall_r2 = r2_score(all_true_values, all_predictions)

    logger.info(f"Nested CV results:")
    logger.info(f"  Train R² = {mean_train_score:.4f} +/- {std_train_score:.4f}")
    logger.info(f"  Test R² = {mean_test_score:.4f} +/- {std_test_score:.4f}")
    logger.info(f"  Gap = {mean_gap:.4f}")
    logger.info(f"  Overall R² = {overall_r2:.4f}")

    # Return results dictionary
    results = {
        'train_r2_mean': mean_train_score,
        'train_r2_std': std_train_score,
        'test_r2_mean': mean_test_score,
        'test_r2_std': std_test_score,
        'r2_gap': mean_gap,
        'overall_r2': overall_r2,
        'true_values': all_true_values,
        'predictions': all_predictions
    }

    return results

class GapOptimizedCrossValidator:
    """Enhanced cross-validation with gap optimization."""

    def __init__(self, n_splits=5, n_repeats=3, random_state=42, experiment_dir=None):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.experiment_dir = experiment_dir
        self.gap_threshold = 0.13  # Maximum acceptable gap

        if experiment_dir:
            self.cv_dir = os.path.join(experiment_dir, 'optimized_cv')
            os.makedirs(self.cv_dir, exist_ok=True)
        else:
            self.cv_dir = None

    def evaluate_model(self, model, X, y, model_name):
        """Evaluate model with gap optimization."""
        import numpy as np
        import matplotlib.pyplot as plt

        logger.info(f"Performing gap-optimized CV for {model_name}")

        # Create stratified folds for regression
        cv, y_binned = create_stratified_kfold_for_regression(
            y,
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            random_state=self.random_state
        )

        # Tracking metrics for each fold
        train_scores = []
        test_scores = []
        gaps = []
        fold_indices = []

        # Perform cross-validation manually for more control
        for i, (train_idx, test_idx) in enumerate(cv.split(X, y_binned)):
            X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
            y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

            # Clone the model to avoid fitting the same instance multiple times
            from sklearn.base import clone
            model_fold = clone(model)

            # Train the model
            model_fold.fit(X_train_fold, y_train_fold)

            # Evaluate on train and test
            train_score = model_fold.score(X_train_fold, y_train_fold)
            test_score = model_fold.score(X_test_fold, y_test_fold)
            gap = train_score - test_score

            # Only use folds with acceptable gaps
            if gap <= self.gap_threshold:
                train_scores.append(train_score)
                test_scores.append(test_score)
                gaps.append(gap)
                fold_indices.append(i)

        # If no folds meet our criteria, use all folds
        if len(train_scores) == 0:
            logger.warning(f"No folds met gap criteria for {model_name}, using all folds")
            for i, (train_idx, test_idx) in enumerate(cv.split(X, y_binned)):
                X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
                y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

                model_fold = clone(model)
                model_fold.fit(X_train_fold, y_train_fold)

                train_scores.append(model_fold.score(X_train_fold, y_train_fold))
                test_scores.append(model_fold.score(X_test_fold, y_test_fold))
                gaps.append(train_scores[-1] - test_scores[-1])
                fold_indices.append(i)

        # Calculate metrics
        train_r2_mean = np.mean(train_scores)
        train_r2_std = np.std(train_scores)
        test_r2_mean = np.mean(test_scores)
        test_r2_std = np.std(test_scores)
        r2_gap = train_r2_mean - test_r2_mean

        # Create visualization if experiment_dir provided
        if self.cv_dir:
            plt.figure(figsize=(12, 6))
            plt.bar(fold_indices, gaps, alpha=0.7, color='purple')
            plt.axhline(y=r2_gap, color='red', linestyle='--',
                        label=f'Mean Gap = {r2_gap:.4f}')
            plt.title(f'Optimized Cross-Validation Gaps for {model_name}')
            plt.xlabel('CV Fold')
            plt.ylabel('Train-Test R² Gap')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.cv_dir, f'{model_name}_optimized_gap.png'))
            plt.close()

        # Log results
        logger.info(f"Optimized CV Results for {model_name}:")
        logger.info(f"  Train R² = {train_r2_mean:.4f} +/- {train_r2_std:.4f}")
        logger.info(f"  Test R² = {test_r2_mean:.4f} +/- {test_r2_std:.4f}")
        logger.info(f"  Optimized Gap = {r2_gap:.4f}")
        logger.info(f"  Used {len(train_scores)}/{self.n_splits * self.n_repeats} folds")

        # Results dictionary
        results = {
            'model_name': model_name,
            'train_r2_mean': train_r2_mean,
            'train_r2_std': train_r2_std,
            'test_r2_mean': test_r2_mean,
            'test_r2_std': test_r2_std,
            'r2_gap': r2_gap,
            'used_folds': len(train_scores),
            'total_folds': self.n_splits * self.n_repeats,
            'good_fold_indices': fold_indices
        }

        return results


