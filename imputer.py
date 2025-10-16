import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from utils import setup_logging

import os
import joblib
import logging
import warnings
import json
from datetime import datetime

# Set up logging
logger = setup_logging()


class RFRegressorWithStd(BaseEstimator, RegressorMixin):
    """Wrapper for RandomForestRegressor that supports return_std parameter."""

    def __init__(self, n_estimators=30, max_depth=5, min_samples_leaf=10,
                 min_samples_split=20, random_state=None):
        # More conservative parameters to reduce overfitting
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            random_state=random_state
        )

    def fit(self, X, y):
        """Fit the model using X, y as training data."""
        self.model.fit(X, y)
        return self

    def predict(self, X, return_std=False):
        """
        Predict using the model.

        Parameters:
        -----------
        X : array-like
            Samples
        return_std : bool, default=False
            If True, return standard deviation of predictions as well

        Returns:
        --------
        array-like
            Predicted values
        """
        y_pred = self.model.predict(X)

        if return_std:
            # Calculate standard deviation from all trees in the forest
            predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
            y_std = np.std(predictions, axis=0)
            return y_pred, y_std

        return y_pred

    @property
    def feature_importances_(self):
        """Return feature importances from the underlying model."""
        return self.model.feature_importances_


def filter_high_missing_columns(X, missing_threshold=0.65):
    """
    Remove columns with missing data above the threshold.

    Parameters:
    -----------
    X : DataFrame
        Input data
    missing_threshold : float, default=0.5
        Threshold for missing data percentage (0.65 = 65%)

    Returns:
    --------
    X_filtered : DataFrame
        Data with high-missing columns removed
    removed_columns : list
        List of removed column names
    """
    missing_percentages = X.isnull().sum() / len(X)
    high_missing_cols = missing_percentages[missing_percentages > missing_threshold].index.tolist()

    if high_missing_cols:
        logger.info(f"Removing {len(high_missing_cols)} columns with >{missing_threshold * 100}% missing data:")
        for col in high_missing_cols:
            logger.info(f"  - {col}: {missing_percentages[col]:.1%} missing")

    X_filtered = X.drop(columns=high_missing_cols)
    return X_filtered, high_missing_cols


def optimize_imputer_iterations_robust(X, missing_threshold=0.65, n_samples=1000,
                                       iteration_range=(5, 10, 15, 20, 25, 30),
                                       n_splits=50, n_seeds=5, validation_split=0.2):
    """
    Optimize the number of iterations for IterativeImputer with robust validation
    and overfitting protection.

    Parameters:
    -----------
    X : DataFrame
        The data with missing values
    missing_threshold : float
        Threshold for removing high-missing columns
    n_samples : int
        Number of samples to use for validation
    iteration_range : tuple
        Range of iteration values to test
    n_splits : int
        Number of cross-validation splits (reduced from 100)
    n_seeds : int
        Number of random seeds to test for stability
    validation_split : float
        Fraction of data to hold out for final validation

    Returns:
    --------
    dict : Results containing best parameters and metrics
    """
    # First, filter out high-missing columns
    X_filtered, removed_cols = filter_high_missing_columns(X, missing_threshold)

    # Get only numeric columns with some complete values for testing
    numeric_cols = X_filtered.select_dtypes(include=['number'])
    complete_cols = numeric_cols.dropna(axis=1)

    if len(complete_cols.columns) == 0:
        logger.warning("No complete numeric columns found after filtering. Using default 10 iterations.")
        return {
            'best_iterations': 10,
            'removed_columns': removed_cols,
            'validation_rmse': None,
            'stability_metrics': None
        }

    # Split data into optimization and hold-out validation sets
    if len(complete_cols) > n_samples:
        complete_cols = complete_cols.sample(n=n_samples, random_state=42)

    train_data, holdout_data = train_test_split(
        complete_cols, test_size=validation_split, random_state=42
    )

    logger.info(f"Using {len(train_data)} samples for optimization, {len(holdout_data)} for validation")
    logger.info(f"Testing {len(iteration_range)} iteration values with {n_seeds} random seeds each")

    # Track results across seeds and iterations
    results = {}

    for n_iter in iteration_range:
        results[n_iter] = {
            'cv_rmses': [],
            'cv_stds': [],
            'convergence_vars': [],
            'holdout_rmses': []
        }

        logger.info(f"\nTesting {n_iter} iterations...")

        for seed in range(n_seeds):
            # Set up cross-validation
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            fold_errors = []

            # Cross-validation on training data
            for train_idx, test_idx in kf.split(train_data):
                X_train_fold = train_data.iloc[train_idx]
                X_test_fold = train_data.iloc[test_idx]

                # Create mask for missing values (adjust percentage based on actual missingness)
                actual_missing_rate = min(0.3, X_filtered.isnull().sum().mean() / len(X_filtered))
                mask = np.random.rand(*X_test_fold.shape) < actual_missing_rate

                # Create copy with masked values set to NaN
                X_test_masked = X_test_fold.copy()
                X_test_masked.values[mask] = np.nan

                # Conservative RF estimator
                rf = RandomForestRegressor(
                    n_estimators=30,
                    max_depth=5,
                    min_samples_leaf=10,
                    min_samples_split=20,
                    random_state=seed
                )

                # Create imputer with early stopping tolerance
                imputer = IterativeImputer(
                    max_iter=n_iter,
                    estimator=rf,
                    random_state=seed,
                    initial_strategy='median',
                    tol=0.01,  # Stricter tolerance for early stopping
                    imputation_order='random'
                )

                try:
                    # Fit on training fold
                    imputer.fit(X_train_fold)

                    # Impute test fold
                    X_test_imputed = imputer.transform(X_test_masked)

                    # Calculate RMSE for masked values
                    if np.sum(mask) > 0:  # Only if there are masked values
                        true_values = X_test_fold.values[mask]
                        imputed_values = X_test_imputed[mask]
                        rmse = np.sqrt(mean_squared_error(true_values, imputed_values))
                        fold_errors.append(rmse)

                except Exception as e:
                    logger.warning(f"Error in fold with seed {seed}, iter {n_iter}: {e}")
                    continue

            if fold_errors:  # Only if we have valid errors
                cv_rmse = np.mean(fold_errors)
                cv_std = np.std(fold_errors)
                results[n_iter]['cv_rmses'].append(cv_rmse)
                results[n_iter]['cv_stds'].append(cv_std)

                # Test on hold-out set for overfitting detection
                try:
                    # Create mask for holdout data
                    holdout_mask = np.random.rand(*holdout_data.shape) < actual_missing_rate
                    holdout_masked = holdout_data.copy()
                    holdout_masked.values[holdout_mask] = np.nan

                    # Fit imputer on full training data
                    imputer_holdout = IterativeImputer(
                        max_iter=n_iter,
                        estimator=RandomForestRegressor(
                            n_estimators=30, max_depth=5, min_samples_leaf=10,
                            min_samples_split=20, random_state=seed
                        ),
                        random_state=seed,
                        initial_strategy='median',
                        tol=0.01
                    )

                    imputer_holdout.fit(train_data)
                    holdout_imputed = imputer_holdout.transform(holdout_masked)

                    if np.sum(holdout_mask) > 0:
                        holdout_true = holdout_data.values[holdout_mask]
                        holdout_pred = holdout_imputed[holdout_mask]
                        holdout_rmse = np.sqrt(mean_squared_error(holdout_true, holdout_pred))
                        results[n_iter]['holdout_rmses'].append(holdout_rmse)

                except Exception as e:
                    logger.warning(f"Error in holdout validation: {e}")

        # Calculate summary statistics for this iteration count
        if results[n_iter]['cv_rmses']:
            mean_cv_rmse = np.mean(results[n_iter]['cv_rmses'])
            std_cv_rmse = np.std(results[n_iter]['cv_rmses'])
            mean_holdout_rmse = np.mean(results[n_iter]['holdout_rmses']) if results[n_iter]['holdout_rmses'] else None

            logger.info(f"  Iterations {n_iter}: CV RMSE = {mean_cv_rmse:.6f} ± {std_cv_rmse:.6f}")
            if mean_holdout_rmse:
                logger.info(f"  Holdout RMSE = {mean_holdout_rmse:.6f}")

    # Select best iteration count with overfitting protection
    best_iter = select_best_iterations(results, iteration_range)

    return {
        'best_iterations': best_iter,
        'removed_columns': removed_cols,
        'detailed_results': results,
        'recommendation': get_selection_reasoning(results, best_iter)
    }


def select_best_iterations(results, iteration_range):
    """
    Select best iteration count considering RMSE, stability, and overfitting.
    """
    scores = {}

    for n_iter in iteration_range:
        if not results[n_iter]['cv_rmses']:
            continue

        # Calculate metrics
        mean_cv_rmse = np.mean(results[n_iter]['cv_rmses'])
        std_cv_rmse = np.std(results[n_iter]['cv_rmses'])
        mean_holdout_rmse = np.mean(results[n_iter]['holdout_rmses']) if results[n_iter][
            'holdout_rmses'] else mean_cv_rmse

        # Stability score (lower variance is better)
        stability_score = 1 / (1 + std_cv_rmse)

        # Overfitting penalty (if holdout RMSE >> CV RMSE)
        overfitting_penalty = max(0, (mean_holdout_rmse - mean_cv_rmse) / mean_cv_rmse)

        # Combined score (lower is better)
        # Balance RMSE performance with stability and overfitting protection
        combined_score = mean_cv_rmse * (1 + overfitting_penalty) / stability_score

        scores[n_iter] = {
            'combined_score': combined_score,
            'cv_rmse': mean_cv_rmse,
            'stability': stability_score,
            'overfitting_penalty': overfitting_penalty
        }

    # Select iteration with best combined score
    best_iter = min(scores.keys(), key=lambda x: scores[x]['combined_score'])

    # Conservative selection: if multiple iterations perform similarly, choose lower
    best_score = scores[best_iter]['combined_score']
    tolerance = 0.01  # 1% tolerance

    for n_iter in sorted(scores.keys()):
        if scores[n_iter]['combined_score'] <= best_score * (1 + tolerance):
            best_iter = n_iter  # Choose the lowest iteration count within tolerance
            break

    logger.info(f"\nSelected {best_iter} iterations:")
    logger.info(f"  CV RMSE: {scores[best_iter]['cv_rmse']:.6f}")
    logger.info(f"  Stability score: {scores[best_iter]['stability']:.4f}")
    logger.info(f"  Overfitting penalty: {scores[best_iter]['overfitting_penalty']:.4f}")

    return best_iter


def get_selection_reasoning(results, best_iter):
    """Generate human-readable reasoning for the selection."""
    if not results[best_iter]['cv_rmses']:
        return "Selection based on default due to insufficient data."

    mean_rmse = np.mean(results[best_iter]['cv_rmses'])
    std_rmse = np.std(results[best_iter]['cv_rmses'])

    reasoning = f"Selected {best_iter} iterations with mean CV RMSE of {mean_rmse:.6f} "
    reasoning += f"and standard deviation of {std_rmse:.6f}. "

    if std_rmse < 0.01:
        reasoning += "Very stable across random seeds. "
    elif std_rmse < 0.05:
        reasoning += "Good stability across random seeds. "
    else:
        reasoning += "Moderate stability across random seeds. "

    return reasoning


def track_imputation_parameters(imputer, optimization_results, experiment_dir, imputation_method):
    """
    Track and save imputation parameters and optimization details.

    Parameters:
    -----------
    imputer : IterativeImputer, SimpleImputer, or FilteringIterativeImputer
        The fitted imputer object
    optimization_results : dict
        Results from the optimization process
    experiment_dir : str
        Directory to save tracking information
    imputation_method : str
        Method used ('simple' or 'iterative')

    Returns:
    --------
    dict : Dictionary containing all tracked parameters
    """
    logger.info("=" * 50)
    logger.info("TRACKING IMPUTATION PARAMETERS...")

    tracking_info = {
        'imputation_method': imputation_method,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'random_states_used': [],
        'iterations_used': None,
        'optimization_details': {},
        'removed_columns': [],
        'filtered_features': [],
        'imputer_type': None,
        'convergence_info': {},
        'estimator_details': {},
        'optimization_seeds_tested': []
    }

    try:
        # Basic imputer information
        tracking_info['imputer_type'] = type(imputer).__name__

        if imputation_method == 'iterative':
            # Handle FilteringIterativeImputer
            if hasattr(imputer, 'removed_columns'):
                tracking_info['removed_columns'] = getattr(imputer, 'removed_columns', [])
                tracking_info['filtered_features'] = getattr(imputer, 'filtered_feature_names', [])
                tracking_info['missing_threshold'] = getattr(imputer, 'missing_threshold', 0.65)

                logger.info(f"Filtering Details:")
                logger.info(f"  Missing threshold: {tracking_info['missing_threshold'] * 100}%")
                logger.info(f"  Removed {len(tracking_info['removed_columns'])} high-missing columns")
                logger.info(f"  Trained on {len(tracking_info['filtered_features'])} features")

                if tracking_info['removed_columns']:
                    logger.info(f"  Removed columns: {tracking_info['removed_columns']}")

                # Get the underlying IterativeImputer
                actual_imputer = getattr(imputer, 'imputer', imputer)

                # Get optimization results from the FilteringIterativeImputer
                if not optimization_results and hasattr(imputer, 'optimization_results'):
                    optimization_results = imputer.optimization_results
            else:
                actual_imputer = imputer

            # Extract IterativeImputer parameters
            tracking_info['max_iter'] = getattr(actual_imputer, 'max_iter', None)
            tracking_info['tol'] = getattr(actual_imputer, 'tol', None)
            tracking_info['imputation_order'] = getattr(actual_imputer, 'imputation_order', None)
            tracking_info['initial_strategy'] = getattr(actual_imputer, 'initial_strategy', None)
            tracking_info['sample_posterior'] = getattr(actual_imputer, 'sample_posterior', None)

            # Get random state from the imputer
            imputer_random_state = getattr(actual_imputer, 'random_state', None)
            if imputer_random_state is not None:
                tracking_info['random_states_used'].append({
                    'component': 'IterativeImputer',
                    'random_state': imputer_random_state,
                    'purpose': 'Main imputation process'
                })

            # Get actual number of iterations used
            if hasattr(actual_imputer, 'n_iter_'):
                tracking_info['iterations_used'] = actual_imputer.n_iter_
                tracking_info['convergence_info']['actual_iterations'] = actual_imputer.n_iter_
                logger.info(f"Actual iterations used: {tracking_info['iterations_used']}")
            else:
                tracking_info['iterations_used'] = tracking_info['max_iter']
                logger.info(f"Max iterations configured: {tracking_info['max_iter']}")

            # Get estimator details
            estimator = getattr(actual_imputer, 'estimator', None)
            if estimator is not None:
                tracking_info['estimator_details']['type'] = type(estimator).__name__
                estimator_random_state = getattr(estimator, 'random_state', None)

                if estimator_random_state is not None:
                    tracking_info['random_states_used'].append({
                        'component': f'Estimator_{tracking_info["estimator_details"]["type"]}',
                        'random_state': estimator_random_state,
                        'purpose': 'Random Forest estimator for imputation'
                    })

                # Get RF parameters if available
                if hasattr(estimator, 'n_estimators'):
                    tracking_info['estimator_details']['parameters'] = {
                        'n_estimators': getattr(estimator, 'n_estimators', None),
                        'max_depth': getattr(estimator, 'max_depth', None),
                        'min_samples_leaf': getattr(estimator, 'min_samples_leaf', None),
                        'min_samples_split': getattr(estimator, 'min_samples_split', None)
                    }

            logger.info(f"Iterative Imputation Configuration:")
            logger.info(f"  Max iterations: {tracking_info['max_iter']}")
            logger.info(f"  Tolerance: {tracking_info['tol']}")
            logger.info(f"  Imputation order: {tracking_info['imputation_order']}")
            logger.info(f"  Initial strategy: {tracking_info['initial_strategy']}")
            logger.info(f"  Sample posterior: {tracking_info['sample_posterior']}")
            logger.info(f"  Estimator: {tracking_info['estimator_details'].get('type', 'Unknown')}")

        elif imputation_method == 'simple':
            tracking_info['strategy'] = getattr(imputer, 'strategy', 'median')
            logger.info(f"Simple Imputation Strategy: {tracking_info['strategy']}")

        # Process optimization results
        if optimization_results:
            tracking_info['optimization_details'] = {
                'best_iterations': optimization_results.get('best_iterations', None),
                'removed_columns_count': len(optimization_results.get('removed_columns', [])),
                'recommendation': optimization_results.get('recommendation', 'N/A'),
                'optimization_used': True
            }

            # Track optimization random states
            tracking_info['optimization_seeds_tested'] = list(range(5))  # We use seeds 0-4
            tracking_info['random_states_used'].append({
                'component': 'Optimization_CV',
                'random_state': 'Multiple (0-4)',
                'purpose': 'Cross-validation for parameter optimization',
                'seeds_tested': tracking_info['optimization_seeds_tested']
            })

            # Add detailed optimization info if available
            if 'detailed_results' in optimization_results:
                detailed = optimization_results['detailed_results']
                iteration_performance = {}

                for iter_count, results in detailed.items():
                    if results.get('cv_rmses'):
                        iteration_performance[str(iter_count)] = {
                            'mean_cv_rmse': np.mean(results['cv_rmses']),
                            'std_cv_rmse': np.std(results['cv_rmses']),
                            'n_seeds_tested': len(results['cv_rmses']),
                            'stability_score': 1 / (1 + np.std(results['cv_rmses']))
                        }

                tracking_info['optimization_details']['iteration_performance'] = iteration_performance
                tracking_info['optimization_details']['total_iterations_tested'] = len(iteration_performance)

            logger.info(f"Optimization Results:")
            logger.info(f"  Selected iterations: {optimization_results.get('best_iterations')}")
            logger.info(f"  Recommendation: {optimization_results.get('recommendation', 'N/A')}")
            logger.info(f"  Seeds tested: {tracking_info['optimization_seeds_tested']}")

        # Log all random states used
        logger.info("Random States Summary:")
        if tracking_info['random_states_used']:
            for i, rs_info in enumerate(tracking_info['random_states_used'], 1):
                logger.info(f"  {i}. {rs_info['component']}: {rs_info['random_state']}")
                logger.info(f"     Purpose: {rs_info['purpose']}")
                if 'seeds_tested' in rs_info:
                    logger.info(f"     Seeds tested: {rs_info['seeds_tested']}")
        else:
            logger.info("  No random states tracked")

        # Add reproducibility information
        main_random_state = None
        if imputation_method == 'iterative':
            if hasattr(imputer, 'random_state'):
                main_random_state = imputer.random_state
            elif hasattr(imputer, 'imputer') and hasattr(imputer.imputer, 'random_state'):
                main_random_state = imputer.imputer.random_state

        tracking_info['reproducibility'] = {
            'can_reproduce': len(tracking_info['random_states_used']) > 0,
            'main_random_state': main_random_state,
            'total_random_states_used': len(tracking_info['random_states_used']),
            'optimization_reproducible': 'optimization_seeds_tested' in tracking_info
        }

    except Exception as e:
        logger.error(f"Error tracking imputation parameters: {str(e)}")
        tracking_info['error'] = str(e)

    # Save tracking information to file
    if experiment_dir:
        tracking_file = os.path.join(experiment_dir, f'imputation_tracking_{imputation_method}.json')
        try:
            with open(tracking_file, 'w') as f:
                json.dump(tracking_info, f, indent=4, default=str)
            logger.info(f"Imputation tracking saved to: {tracking_file}")
        except Exception as e:
            logger.error(f"Error saving tracking file: {str(e)}")

    logger.info("=" * 50)
    return tracking_info


class FilteringIterativeImputer:
    """
    Wrapper for IterativeImputer that handles high-missing column filtering consistently.

    This ensures that the same columns are filtered during both fit and transform operations.
    """

    def __init__(self, missing_threshold=0.65, random_state=42):
        self.missing_threshold = missing_threshold
        self.random_state = random_state
        self.removed_columns = None
        self.filtered_feature_names = None
        self.imputer = None
        self.is_fitted = False
        self.optimization_results = None  # Store optimization results for tracking

    def fit(self, X, y=None):
        """Fit the imputer with automatic high-missing column filtering."""
        logger.info("Fitting FilteringIterativeImputer...")

        # Filter high-missing columns
        X_filtered, self.removed_columns = filter_high_missing_columns(X, self.missing_threshold)
        self.filtered_feature_names = X_filtered.columns.tolist()

        logger.info(f"Filtered {len(self.removed_columns)} high-missing columns")
        logger.info(f"Training imputer on {len(self.filtered_feature_names)} remaining features")

        # Get optimization results
        self.optimization_results = optimize_imputer_iterations_robust(
            X_filtered,
            missing_threshold=self.missing_threshold
        )

        optimal_iterations = self.optimization_results['best_iterations']
        logger.info(f"Using {optimal_iterations} iterations for imputation")

        # Create and fit the actual imputer
        rf_estimator = RFRegressorWithStd(
            n_estimators=30,
            max_depth=5,
            min_samples_leaf=10,
            min_samples_split=20,
            random_state=self.random_state
        )

        #self.imputer = IterativeImputer(
            #estimator=rf_estimator,
            #max_iter=optimal_iterations,
            #tol=0.01,
            #imputation_order='random',
            #sample_posterior=False,
            #n_nearest_features=None,
            #skip_complete=True,
            #random_state=self.random_state,
            #initial_strategy='median',
            #verbose=1
        #)
        self.imputer = IterativeImputer(
            estimator=rf_estimator,
            max_iter=optimal_iterations,
            tol=0.01,
            imputation_order='ascending',
            min_value=X_filtered[self.filtered_feature_names].min(numeric_only=True),
            max_value=X_filtered[self.filtered_feature_names].max(numeric_only=True),
            sample_posterior=False,
            n_nearest_features=None,
            skip_complete=True,
            random_state=self.random_state,
            initial_strategy='median',
            verbose=1
        )

        # Fit on filtered data
        self.imputer.fit(X_filtered)
        self.is_fitted = True

        return self

    def transform(self, X):
        """Transform data with consistent column filtering."""
        if not self.is_fitted:
            raise ValueError("FilteringIterativeImputer must be fitted before transform")

        # Apply the same filtering as during fit
        X_filtered = apply_same_filtering(X, self.removed_columns)

        # Ensure we have the same columns as during training
        missing_cols = [col for col in self.filtered_feature_names if col not in X_filtered.columns]
        if missing_cols:
            logger.warning(f"Missing columns in transform data: {missing_cols}")
            # Add missing columns with NaN values
            for col in missing_cols:
                X_filtered[col] = np.nan

        # Reorder to match training order
        X_filtered = X_filtered[self.filtered_feature_names]

        # Transform using the fitted imputer
        X_imputed = self.imputer.transform(X_filtered)

        # Return as DataFrame with correct column names
        return pd.DataFrame(X_imputed, columns=self.filtered_feature_names, index=X.index)

    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def get_tracking_info(self, experiment_dir=None):
        """
        Get comprehensive tracking information about this imputer.

        Returns:
        --------
        dict : Complete tracking information
        """
        if not self.is_fitted:
            logger.warning("Imputer not fitted yet - tracking info may be incomplete")

        tracking_info = track_imputation_parameters(
            imputer=self,
            optimization_results=self.optimization_results,
            experiment_dir=experiment_dir,
            imputation_method='iterative'
        )

        return tracking_info


def create_iterative_imputer_with_filtering(X_outlier_handled, missing_threshold=0.65, random_state=42,
                                            experiment_dir=None):
    """
    Create and fit a FilteringIterativeImputer with tracking.

    Parameters:
    -----------
    X_outlier_handled : DataFrame
        Data after outlier handling
    missing_threshold : float
        Threshold for removing high-missing columns
    random_state : int, default=42
        Random seed for reproducibility
    experiment_dir : str, optional
        Directory to save tracking information

    Returns:
    --------
    tuple: (imputer, removed_columns, filtered_feature_names, tracking_info)
        - imputer: The fitted FilteringIterativeImputer
        - removed_columns: List of column names that were removed
        - filtered_feature_names: List of remaining column names
        - tracking_info: Dictionary with tracking information
    """
    logger.info("Creating and fitting FilteringIterativeImputer...")

    # Create the filtering imputer
    imputer = FilteringIterativeImputer(missing_threshold, random_state)

    # Fit it to get the filtering and optimization information
    imputer.fit(X_outlier_handled)

    # Extract the information
    removed_columns = imputer.removed_columns
    filtered_feature_names = imputer.filtered_feature_names

    # Get comprehensive tracking information
    tracking_info = imputer.get_tracking_info(experiment_dir=experiment_dir)

    logger.info(f"SUCCESS: Created imputer with {len(filtered_feature_names)} features")
    logger.info(f"Removed {len(removed_columns)} high-missing columns")

    return imputer, removed_columns, filtered_feature_names, tracking_info


def create_simple_imputer_with_filtering(X_outlier_handled, missing_threshold=0.65,
                                         strategy='median', experiment_dir=None):
    """
    Create a SimpleImputer with column filtering and tracking capabilities.

    Parameters:
    -----------
    X_outlier_handled : DataFrame
        Data after outlier handling
    missing_threshold : float, default=0.50
        Proportion of missing data above which columns are removed.
        E.g., 0.50 means columns with >50% missing data are dropped.
    strategy : str, default='median'
        Imputation strategy: 'mean', 'median', 'most_frequent', or 'constant'
    experiment_dir : str, optional
        Directory to save tracking information

    Returns:
    --------
    tuple: (imputer, removed_columns, filtered_feature_names, tracking_info)
        - imputer: The configured SimpleImputer (fitted on filtered data)
        - removed_columns: List of column names that were removed
        - filtered_feature_names: List of remaining column names
        - tracking_info: Dictionary with tracking information
    """

    from pathlib import Path

    logger.info(f"Creating SimpleImputer with filtering (threshold={missing_threshold})...")

    # Calculate missing percentage for each column
    missing_percentages = (X_outlier_handled.isnull().sum() / len(X_outlier_handled)).to_dict()

    # Identify columns to remove
    removed_columns = [
        col for col, missing_pct in missing_percentages.items()
        if missing_pct > missing_threshold
    ]

    # Get filtered feature names
    filtered_feature_names = [
        col for col in X_outlier_handled.columns
        if col not in removed_columns
    ]

    logger.info(f"Filtering columns with >{missing_threshold * 100}% missing data")
    logger.info(f"Removed {len(removed_columns)} columns: {removed_columns}")
    logger.info(f"Retained {len(filtered_feature_names)} columns for imputation")

    # Filter the data
    X_filtered = X_outlier_handled[filtered_feature_names]

    # Create and fit SimpleImputer on filtered data
    imputer = SimpleImputer(strategy=strategy)
    imputer.fit(X_filtered)

    # Create tracking info
    tracking_info = {
        'imputation_method': 'simple_with_filtering',
        'strategy': strategy,
        'missing_threshold': missing_threshold,
        'total_original_columns': len(missing_percentages),
        'removed_columns_count': len(removed_columns),
        'retained_columns_count': len(filtered_feature_names),
        'removed_columns': removed_columns,
        'filtered_feature_names': filtered_feature_names,
        'missing_percentages': {
            col: f"{pct:.2%}" for col, pct in missing_percentages.items()
        },
        'removed_columns_details': {
            col: f"{missing_percentages[col]:.2%}"
            for col in removed_columns
        }
    }

    # Save to file if directory provided
    if experiment_dir:
        experiment_path = Path(experiment_dir)
        experiment_path.mkdir(parents=True, exist_ok=True)

        tracking_file = experiment_path / 'simple_imputer_tracking_info.json'
        with open(tracking_file, 'w') as f:
            json.dump(tracking_info, f, indent=2)
        logger.info(f"Saved tracking info to {tracking_file}")

    # Also call your existing tracking function if it exists
    try:
        additional_tracking = track_imputation_parameters(
            imputer=imputer,
            optimization_results=None,
            experiment_dir=experiment_dir,
            imputation_method='simple_with_filtering'
        )
        tracking_info['imputation_parameters'] = additional_tracking
    except NameError:
        # track_imputation_parameters doesn't exist, skip it
        pass

    logger.info(f"SUCCESS: Created imputer with {len(filtered_feature_names)} features")

    return imputer, removed_columns, filtered_feature_names, tracking_info


def apply_same_filtering(X, removed_columns):
    """
    Apply the same column filtering that was used during training.

    Parameters:
    -----------
    X : DataFrame
        Data to filter
    removed_columns : list
        List of column names to remove

    Returns:
    --------
    DataFrame
        Filtered data with high-missing columns removed
    """
    if not removed_columns:
        return X

    # Remove the same columns that were removed during training
    columns_to_remove = [col for col in removed_columns if col in X.columns]

    if columns_to_remove:
        logger.info(f"Removing {len(columns_to_remove)} high-missing columns for consistency with training")
        X_filtered = X.drop(columns=columns_to_remove)
    else:
        X_filtered = X

    return X_filtered