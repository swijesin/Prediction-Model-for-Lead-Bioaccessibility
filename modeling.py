import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RepeatedKFold,RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict
from scipy.stats import uniform, loguniform
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Import the EnhancedCrossValidator
from cross_validation import (GapOptimizedCrossValidator,
                              get_optimal_cv_params, create_stratified_kfold_for_regression)

# Try to import shap for model interpretability
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModelWrapper(BaseEstimator, RegressorMixin):
    """A wrapper class that applies feature engineering before prediction."""

    def __init__(self, model, feature_engineer):
        self.model = model
        self.feature_engineer = feature_engineer

    def fit(self, X, y):
        """Fit the model to data X and target y."""
        if self.feature_engineer is not None:
            X_transformed = self.feature_engineer.transform(X)
            self.model.fit(X_transformed, y)
        else:
            self.model.fit(X, y)
        return self

    def predict(self, X):
        """Predict target for X."""
        if self.feature_engineer is not None:
            X_transformed = self.feature_engineer.transform(X)
            return self.model.predict(X_transformed)
        return self.model.predict(X)

    @property
    def feature_importances_(self):
        """Get feature importances from the wrapped model."""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        raise AttributeError("The wrapped model has no feature_importances_")

def get_optimal_cv_params_additions():
    """
    Add these to your existing get_optimal_cv_params function
    """
    cv_params_additions = {
        'SVR': {
            'n_splits': 5,
            'n_repeats': 2,  # SVR can be computationally expensive
            'random_state': 42,
            'gap_threshold': 0.15
        },
        'svr': {  # lowercase version
            'n_splits': 5,
            'n_repeats': 2,
            'random_state': 42,
            'gap_threshold': 0.15
        },
        'KNN': {
            'n_splits': 5,
            'n_repeats': 3,
            'random_state': 42,
            'gap_threshold': 0.10  # KNN typically has lower overfitting
        },
        'knn': {  # lowercase version
            'n_splits': 5,
            'n_repeats': 3,
            'random_state': 42,
            'gap_threshold': 0.10
        }
    }
    return cv_params_additions

def augment_data_for_nn(X, y, n_samples=100):
    """
    Create synthetic samples using a regression-appropriate technique.

    Parameters:
    -----------
    X : DataFrame
        Feature matrix
    y : Series
        Target values
    n_samples : int, default=100
        Number of synthetic samples to generate

    Returns:
    --------
    X_augmented : DataFrame
        Original features plus synthetic samples
    y_augmented : Series
        Original targets plus synthetic targets
    """

    from sklearn.neighbors import NearestNeighbors

    # Ensure we have enough data to work with
    if len(X) < 5:
        print("Not enough samples for augmentation. Need at least 5 samples.")
        return X, y

    # Make sure n_samples is reasonable (generate at least 20 samples)
    n_samples = max(n_samples, 20)

    # Initialize arrays for synthetic data
    synthetic_X = []
    synthetic_y = []

    # Find k nearest neighbors for each sample
    k = min(5, len(X) - 1)  # Ensure k is less than the number of samples
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)  # +1 because it includes the point itself
    distances, indices = nn.kneighbors(X)

    # Generate synthetic samples
    for _ in range(n_samples):
        # Choose a random sample
        idx = np.random.randint(0, len(X))
        sample_X = X.iloc[idx].values
        sample_y = y.iloc[idx]

        # Get one of its neighbors (skip the first as it's the point itself)
        neighbor_idx = indices[idx][np.random.randint(1, k + 1)]
        neighbor_X = X.iloc[neighbor_idx].values
        neighbor_y = y.iloc[neighbor_idx]

        # Generate a random interpolation factor
        alpha = np.random.random()

        # Create synthetic sample through interpolation
        new_sample_X = sample_X + alpha * (neighbor_X - sample_X)
        new_sample_y = sample_y + alpha * (neighbor_y - sample_y)

        synthetic_X.append(new_sample_X)
        synthetic_y.append(new_sample_y)

    # Convert to DataFrame and Series
    synthetic_X_df = pd.DataFrame(synthetic_X, columns=X.columns)
    synthetic_y_series = pd.Series(synthetic_y, name=y.name)

    # Combine with original data
    X_augmented = pd.concat([X, synthetic_X_df], ignore_index=True)
    y_augmented = pd.concat([y, synthetic_y_series], ignore_index=True)

    print(
        f"Data augmentation complete. Original samples: {len(X)}, Synthetic samples: {len(synthetic_X_df)}, Total: {len(X_augmented)}")

    return X_augmented, y_augmented


class ModelTrainer:
    """Class for training and tuning machine learning models with optimized CV."""

    def __init__(self, experiment_dir, random_state=42, n_jobs=-1):
        self.experiment_dir = experiment_dir
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.models_dir = os.path.join(experiment_dir, 'models')
        os.makedirs(self.models_dir, exist_ok=True)

        # Initialize optimized cross-validator
        self.cross_validator = GapOptimizedCrossValidator(
            n_splits=5,
            n_repeats=3,
            random_state=random_state,
            experiment_dir=experiment_dir
        )

    def train_models(self, X, y, models=None, use_leak_free=True):
        """
        Train multiple regression models with optimized CV.

        Parameters:
        -----------
        X : DataFrame
            Feature matrix
        y : Series
            Target values
        models : list or None
            List of model types to train. If None, trains all available models.
        use_leak_free : bool, default=True
            Whether to use leak-free training approach

        Returns:
        --------
        dict
            Dictionary of trained models
        """
        available_models = {
            'random_forest': self._train_random_forest,
            'hist_gradient_boosting': self._train_hist_gradient_boosting,
            'xgboost': self._train_xgboost,
            'lightgbm': self._train_lightgbm,
            'neural_network': self.train_neural_network_with_augmentation,
            'svr': self._train_svr,  # Add SVR
            'knn': self._train_knn  # Add KNN
        }

        if models is None:
            models = list(available_models.keys())

        trained_models = {}
        cv_results = {}

        # Print the models being trained
        approach = "leak-free" if use_leak_free else "optimized"
        logger.info(f"Training models with {approach} CV: {models}")

        if use_leak_free and hasattr(self, 'train_models_leak_free'):
            # Use leak-free approach if available and requested
            logger.info("Using leak-free cross-validation approach")

            # Define model creators for leak-free training
            model_creators = {}
            for model_name in models:
                if model_name == 'random_forest':
                    def create_random_forest():
                        from sklearn.ensemble import RandomForestRegressor
                        return RandomForestRegressor(n_estimators=100, random_state=self.random_state)

                    model_creators['random_forest'] = create_random_forest
                elif model_name == 'hist_gradient_boosting':
                    def create_hist_gradient_boosting():
                        from sklearn.ensemble import HistGradientBoostingRegressor
                        return HistGradientBoostingRegressor(random_state=self.random_state)

                    model_creators['hist_gradient_boosting'] = create_hist_gradient_boosting
                elif model_name == 'xgboost':
                    def create_xgboost():
                        import xgboost as xgb
                        return xgb.XGBRegressor(random_state=self.random_state)

                    model_creators['xgboost'] = create_xgboost
                elif model_name == 'lightgbm':
                    def create_lightgbm():
                        import lightgbm as lgb
                        return lgb.LGBMRegressor(random_state=self.random_state)

                    model_creators['lightgbm'] = create_lightgbm
                elif model_name == 'neural_network':
                    def create_neural_network():
                        from sklearn.neural_network import MLPRegressor
                        return MLPRegressor(
                            hidden_layer_sizes=(64, 32),
                            activation='relu',
                            solver='adam',
                            alpha=0.001,
                            batch_size=32,
                            learning_rate_init=0.001,
                            max_iter=1000,
                            early_stopping=True,
                            random_state=self.random_state
                        )

                    model_creators['neural_network'] = create_neural_network
                elif model_name == 'svr':
                    def create_svr():
                        from sklearn.svm import SVR
                        from sklearn.preprocessing import StandardScaler
                        from sklearn.pipeline import Pipeline
                        return Pipeline([
                            ('scaler', StandardScaler()),
                            ('svr', SVR(C=1.0, gamma='scale', epsilon=0.1, kernel='rbf'))
                        ])

                    model_creators['svr'] = create_svr
                elif model_name == 'knn':
                    def create_knn():
                        from sklearn.neighbors import KNeighborsRegressor
                        return KNeighborsRegressor(n_neighbors=5, weights='uniform')
                    model_creators['knn'] = create_knn

            # Call train_models_leak_free method with the model creators
            # This will handle the leak-free training behind the scenes
            trained_models = self.train_models_leak_free(X, y, model_creators)

            # Extract CV results
            cv_results = getattr(self, 'cv_results', {})

        else:
            # Traditional approach
            logger.info("Using traditional cross-validation approach")

            for model_name in models:
                if model_name in available_models:
                    logger.info(f"Training {model_name} model...")

                    # Train the model with model-specific CV parameters
                    model = available_models[model_name](X, y)
                    trained_models[model_name] = model

                    # Evaluate with optimized cross-validation
                    if hasattr(self, 'cross_validator') and self.cross_validator is not None:
                        cv_result = self.cross_validator.evaluate_model(model, X, y, model_name)
                        cv_results[model_name] = cv_result

                        # Save CV results
                        cv_results_path = os.path.join(self.models_dir, f"{model_name}_cv_results.pkl")
                        joblib.dump(cv_result, cv_results_path)

                    # Save the model
                    joblib.dump(model, os.path.join(self.models_dir, f"{model_name}.pkl"))

                    logger.info(f"Finished training {model_name}")

        # Create optimized weighted ensemble if we have multiple models
        if len(trained_models) > 1:
            logger.info("Creating optimized weighted ensemble...")
            try:
                ensemble_model = self.create_optimized_weighted_ensemble(trained_models, X, y)
                trained_models['optimized_weighted_ensemble'] = ensemble_model

                # Save the ensemble model
                joblib.dump(ensemble_model, os.path.join(self.models_dir, "optimized_weighted_ensemble.pkl"))
            except Exception as e:
                logger.error(f"Error creating optimized weighted ensemble: {str(e)}")
                logger.warning("Proceeding without optimized weighted ensemble")

        # Store CV results
        self.cv_results = cv_results

        return trained_models

    def train_models_leak_free(self, X, y, model_creators):
        """
        Train multiple regression models with leak-free cross-validation.

        Parameters:
        -----------
        X : DataFrame
            Feature matrix
        y : Series
            Target values
        model_creators : dict
            Dictionary of model creator functions

        Returns:
        --------
        dict
            Dictionary of trained models
        """
        from sklearn.model_selection import KFold
        import numpy as np
        from sklearn.metrics import r2_score
        import pandas as pd

        trained_models = {}
        cv_results = {}

        # For each model type
        for model_name, model_creator in model_creators.items():
            logger.info(f"Training {model_name} with leak-free CV...")

            # Create cross-validation folds
            cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)

            # Arrays to store fold results
            train_scores = []
            test_scores = []
            all_true_values = []
            all_predictions = []

            # For each fold
            for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
                # Split data for this fold
                X_train_fold = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
                X_test_fold = X.iloc[test_idx] if hasattr(X, 'iloc') else X[test_idx]
                y_train_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
                y_test_fold = y.iloc[test_idx] if hasattr(y, 'iloc') else y[test_idx]

                # Create and train model for this fold
                fold_model = model_creator()
                fold_model.fit(X_train_fold, y_train_fold)

                # Evaluate on train and test sets
                train_score = fold_model.score(X_train_fold, y_train_fold)
                test_score = fold_model.score(X_test_fold, y_test_fold)

                # Store scores
                train_scores.append(train_score)
                test_scores.append(test_score)

                # Store predictions for overall R²
                all_true_values.extend(y_test_fold)
                all_predictions.extend(fold_model.predict(X_test_fold))

                logger.info(f"  Fold {fold_idx + 1}: Train R² = {train_score:.4f}, Test R² = {test_score:.4f}")

            # Calculate metrics
            train_r2_mean = np.mean(train_scores)
            train_r2_std = np.std(train_scores)
            test_r2_mean = np.mean(test_scores)
            test_r2_std = np.std(test_scores)
            r2_gap = train_r2_mean - test_r2_mean
            overall_r2 = r2_score(all_true_values, all_predictions)

            logger.info(f"  Mean Train R² = {train_r2_mean:.4f} +/- {train_r2_std:.4f}")
            logger.info(f"  Mean Test R² = {test_r2_mean:.4f} +/- {test_r2_std:.4f}")
            logger.info(f"  R² Gap = {r2_gap:.4f}")
            logger.info(f"  Overall R² = {overall_r2:.4f}")

            # Store CV results
            cv_results[model_name] = {
                'train_r2_mean': train_r2_mean,
                'train_r2_std': train_r2_std,
                'test_r2_mean': test_r2_mean,
                'test_r2_std': test_r2_std,
                'r2_gap': r2_gap,
                'overall_r2': overall_r2,
                'true_values': all_true_values,
                'predictions': all_predictions
            }

            # Train final model on all data
            final_model = model_creator()
            final_model.fit(X, y)
            trained_models[model_name] = final_model

        # Store CV results for use later
        self.cv_results = cv_results

        return trained_models

    def train_with_optimized_cv(self, model_class, param_grid, X, y, model_type, n_iter=10):
        """
        Train a model with optimized cross-validation parameters.

        Parameters:
        -----------
        model_class : estimator class
            Model class to instantiate
        param_grid : dict
            Hyperparameter grid
        X : DataFrame
            Features
        y : Series
            Target
        model_type : str
            Model type name for CV optimization
        n_iter : int, default=10
            Number of parameter settings to try

        Returns:
        --------
        best_model : estimator
            Trained model
        """

        # Get optimal CV parameters for this model type
        cv_params = get_optimal_cv_params(model_type)

        # Create stratified folds for regression
        cv, y_binned = create_stratified_kfold_for_regression(
            y,
            n_splits=cv_params['n_splits'],
            n_repeats=cv_params['n_repeats'],
            random_state=cv_params['random_state']
        )

        # Create base model
        base_model = model_class(random_state=self.random_state)

        # Create search
        search = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=n_iter,
            cv=list(cv.split(X, y_binned))[:cv_params['n_splits']],
            scoring='r2',
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=1,
            return_train_score=True
        )

        # Fit search
        search.fit(X, y)

        # Get best model
        best_model = search.best_estimator_

        # Log results
        logger.info(f"Best parameters for {model_type}: {search.best_params_}")
        logger.info(f"Best CV score: {search.best_score_:.4f}")

        return best_model
    def _create_cv_summary(self, cv_results):
        """Create summary of cross-validation results."""
        cv_summary_dir = os.path.join(self.experiment_dir, 'cv_summary')
        os.makedirs(cv_summary_dir, exist_ok=True)

        # Extract key metrics
        summary_data = []
        for model_name, result in cv_results.items():
            summary_data.append({
                'Model': model_name,
                'CV Train R²': f"{result['train_r2_mean']:.4f} +/- {result['train_r2_std']:.4f}",
                'CV Test R²': f"{result['test_r2_mean']:.4f} +/- {result['test_r2_std']:.4f}",
                'Overfitting Gap': f"{result['r2_gap']:.4f}",
                'Train R² Mean': result['train_r2_mean'],
                'Test R² Mean': result['test_r2_mean'],
                'Gap': result['r2_gap']
            })

        # Create DataFrame and save to CSV
        cv_summary_df = pd.DataFrame(summary_data)
        cv_summary_df.to_csv(os.path.join(cv_summary_dir, 'cv_summary.csv'), index=False)

        # Create visualization
        plt.figure(figsize=(14, 8))

        # Sort models by CV Test R²
        cv_summary_df = cv_summary_df.sort_values('Test R² Mean', ascending=False)

        models = cv_summary_df['Model']
        train_r2 = cv_summary_df['Train R² Mean']
        test_r2 = cv_summary_df['Test R² Mean']
        gaps = cv_summary_df['Gap']

        # Set up bar positions
        x = np.arange(len(models))
        width = 0.25

        # Create bars
        plt.bar(x - width, train_r2, width, label='CV Train R²', color='blue', alpha=0.7)
        plt.bar(x, test_r2, width, label='CV Test R²', color='green', alpha=0.7)
        plt.bar(x + width, gaps, width, label='Overfitting Gap', color='red', alpha=0.7)

        # Add labels
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Cross-Validation Results Comparison')
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
        plt.savefig(os.path.join(cv_summary_dir, 'cv_comparison.png'))
        plt.close()

        logger.info(f"Cross-validation summary saved to {cv_summary_dir}")

    def _train_random_forest(self, X, y):
        """Train a Random Forest Regressor with hyperparameter tuning and cross-validation."""
        param_grid = {
            'n_estimators': [100, 300, 500],
            'max_depth': [20, 30, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.7]
        }


        # Train with CV-based hyperparameter optimization
        best_model = self.train_with_optimized_cv(
            RandomForestRegressor,
            param_grid,
            X,
            y,
            'random_forest',
            n_iter=20
        )


        logger.info(f"Best Random Forest parameters: {best_model.get_params()}")

        return best_model

    def _train_hist_gradient_boosting(self, X, y):
        """Train a Histogram-based Gradient Boosting Regressor with cross-validation."""
        param_grid = {
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [10, 20, None],
            'max_iter': [100, 200, 500],
            'min_samples_leaf': [1, 5, 20],
            'l2_regularization': [0, 0.01, 0.1]
        }

        # Train with CV-based hyperparameter optimization
        best_model = self.train_with_optimized_cv(
            HistGradientBoostingRegressor,
            param_grid,
            X, y,
            'HistGradientBoosting',
            n_iter=15
        )

        logger.info(f"Best Histogram GBM parameters: {best_model.get_params()}")

        return best_model

    def _train_xgboost(self, X, y):
        """Train an XGBoost Regressor with cross-validation."""
        param_grid = {
            'n_estimators': [100, 300],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 6],
            'min_child_weight': [1, 3],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }


        # Train with CV-based hyperparameter optimization
        best_model = self.train_with_optimized_cv(
            xgb.XGBRegressor,
            param_grid,
            X, y,
            'XGBoost',
            n_iter=10
        )

        logger.info(f"Best XGBoost parameters: {best_model.get_params()}")

        return best_model

    def _train_lightgbm(self, X, y):
        """Train a LightGBM Regressor with cross-validation."""
        param_grid = {
            'n_estimators': [100, 300],
            'learning_rate': [0.01, 0.1],
            'max_depth': [5, 7],
            'num_leaves': [31, 63],
            'min_child_samples': [10, 20],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }

        # Train with CV-based hyperparameter optimization
        best_model = self.train_with_optimized_cv(
            lgb.LGBMRegressor,
            param_grid,
            X, y,
            'LightGBM',
            n_iter=8
        )

        logger.info(f"Best LightGBM parameters: {best_model.get_params()}")

        return best_model

    def _train_svr(self, X, y):
        """Train a Support Vector Regressor with RBF kernel and hyperparameter tuning."""
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        # SVR requires feature scaling for optimal performance
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR())
        ])

        param_grid = {
            'svr__C': [0.1, 1, 10, 100],
            'svr__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'svr__epsilon': [0.01, 0.1, 0.2, 0.5],
            'svr__kernel': ['rbf']  # Focus on RBF kernel as requested
        }

        # Train with CV-based hyperparameter optimization
        best_model = self.train_with_optimized_cv(
            lambda **kwargs: Pipeline([
                ('scaler', StandardScaler()),
                ('svr', SVR(**kwargs))
            ]),
            param_grid,
            X, y,
            'SVR',
            n_iter=20
        )

        logger.info(f"Best SVR parameters: {best_model.get_params()}")

        return best_model

    def _train_knn(self, X, y):
        """Train a k-Nearest Neighbors Regressor with hyperparameter tuning."""
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11, 15, 20],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree'],
            'leaf_size': [20, 30, 40, 50],
            'p': [1, 2]  # 1 for Manhattan distance, 2 for Euclidean
        }

        # Train with CV-based hyperparameter optimization
        best_model = self.train_with_optimized_cv(
            KNeighborsRegressor,
            param_grid,
            X, y,
            'KNN',
            n_iter=15
        )

        logger.info(f"Best KNN parameters: {best_model.get_params()}")

        return best_model

    def train_neural_network_with_augmentation(self, X, y):
        """Train a neural network with controlled complexity and gap-optimized cross-validation."""

        logger.info("Training neural network with controlled complexity and gap-optimized CV...")

        # Get neural network specific CV parameters
        cv_params = get_optimal_cv_params('neural_network')

        # Create stratified folds for regression
        cv_obj, y_binned = create_stratified_kfold_for_regression(
            y,
            n_splits=cv_params['n_splits'],
            n_repeats=cv_params['n_repeats'],
            random_state=self.random_state
        )

        # Create the CV indices for search
        cv_indices = list(cv_obj.split(X, y_binned))

        # Create a robust pipeline that includes preprocessing
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('nn', MLPRegressor(random_state=self.random_state))
        ])

        # Simplified parameter grid focused on stability
        param_grid = {
            'nn__hidden_layer_sizes': [
                (32, 16),  # Simple architecture
                (64, 32),  # Medium architecture
                (64, 32, 16),  # Deeper architecture
                (32, 32, 32)  # Equal-width architecture
            ],
            'nn__activation': ['relu', 'tanh'],
            'nn__alpha': [0.001, 0.01, 0.1, 1.0],  # Higher regularization
            'nn__learning_rate': ['constant', 'adaptive'],
            'nn__learning_rate_init': [0.001, 0.005, 0.01],
            'nn__max_iter': [1000],
            'nn__early_stopping': [True],
            'nn__validation_fraction': [0.2],
            'nn__n_iter_no_change': [20],
            'nn__solver': ['adam']
        }

        # Use a smaller augmentation approach
        # Only augment if the dataset is small
        if len(X) < 100:
            X_aug, y_aug = augment_data_for_nn(X, y, n_samples=min(100, len(X)))
        else:
            X_aug, y_aug = X, y

        try:
            # Train using gap-optimized cross-validation

            search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_grid,
                n_iter=15,
                cv=cv_indices,  # Use the stratified CV indices
                scoring='r2',
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=1,
                return_train_score=True,
                error_score='raise'
            )

            search.fit(X_aug, y_aug)

            # Log detailed results
            logger.info(f"Best Neural Network parameters: {search.best_params_}")
            logger.info(f"Best score: {search.best_score_:.4f}")

            # Check for large gaps in cross-validation results
            train_scores = search.cv_results_['mean_train_score']
            test_scores = search.cv_results_['mean_test_score']
            gaps = train_scores - test_scores

            # Log CV gap information
            logger.info(f"Mean CV gap: {np.mean(gaps):.4f}")
            logger.info(f"Max CV gap: {np.max(gaps):.4f}")

            # Return the best model
            best_model = search.best_estimator_

        except Exception as e:
            logger.error(f"Error during neural network training: {str(e)}")
            logger.warning("Falling back to a simple neural network with default parameters")

            # Fallback to a simple model with high regularization
            fallback_model = Pipeline([
                ('scaler', StandardScaler()),
                ('nn', MLPRegressor(
                    hidden_layer_sizes=(32, 16),
                    alpha=0.1,  # High regularization
                    max_iter=1000,
                    early_stopping=True,
                    validation_fraction=0.2,
                    random_state=self.random_state
                ))
            ])

            fallback_model.fit(X, y)
            best_model = fallback_model

        return best_model



    def create_ensemble(self, models, X, y):
        """Create a voting ensemble from multiple models with cross-validation evaluation."""
        # Filter out models that shouldn't be included in the ensemble
        ensemble_models = []

        for name, model in models.items():
            if name not in ['neural_network']:  # exclude some models
                ensemble_models.append((name, model))

        if len(ensemble_models) < 2:
            logger.warning("Not enough models for ensemble, returning best model")
            return max(models.items(), key=lambda x: self._evaluate_model(x[1], X, y)[0])[1]

        # Create and fit the ensemble
        ensemble = VotingRegressor(ensemble_models)
        ensemble.fit(X, y)

        # Evaluate ensemble with cross-validation
        if hasattr(self, 'cross_validator'):
            logger.info("Evaluating ensemble with cross-validation...")
            cv_result = self.cross_validator.evaluate_model(ensemble, X, y, 'voting_ensemble')

            # Save CV results
            if hasattr(self, 'cv_results'):
                self.cv_results['voting_ensemble'] = cv_result

                # Update CV summary
                self._create_cv_summary(self.cv_results)

            cv_results_path = os.path.join(self.models_dir, "voting_ensemble_cv_results.pkl")
            joblib.dump(cv_result, cv_results_path)

        # Save the ensemble
        joblib.dump(ensemble, os.path.join(self.models_dir, "ensemble.pkl"))

        return ensemble

    def create_optimized_weighted_ensemble(self, models, X, y):
        """
        Create an ensemble with optimally weighted models using L1 regularization.

        Parameters:
        -----------
        models : dict
            Dictionary of trained models {name: model}
        X : DataFrame
            Feature matrix for training the ensemble weights
        y : Series
            Target values

        Returns:
        --------
        ensemble : VotingRegressor
            Trained ensemble model with optimized weights
        """

        logger.info("Creating optimized weighted ensemble with L1 regularization...")

        # Filter out any existing ensemble models
        base_models = {name: model for name, model in models.items()
                       if 'ensemble' not in name.lower()}

        if len(base_models) < 2:
            logger.warning("Not enough base models for ensemble. Need at least 2.")
            # Return the single model if there's only one
            if len(base_models) == 1:
                name, model = list(base_models.items())[0]
                logger.info(f"Returning single model {name} as ensemble")
                return model
            else:
                raise ValueError("No base models available for ensemble")

        # Generate predictions from all base models
        predictions = {}
        for name, model in base_models.items():
            predictions[name] = model.predict(X)

        # Create prediction matrix
        pred_df = pd.DataFrame(predictions)
        base_models_names = list(pred_df.columns)

        # Use LassoCV to find optimal weights
        meta_model = LassoCV(
            cv=5,
            max_iter=10000,
            tol=1e-4,
            n_jobs=-1,
            random_state=42
        )

        # Fit the meta-model to find optimal weights
        meta_model.fit(pred_df, y)

        # Get the weights for each model
        weights = meta_model.coef_

        # Find which models have non-zero weights
        selected_indices = np.where(weights != 0)[0]
        selected_models = [base_models_names[i] for i in selected_indices]
        selected_weights = weights[selected_indices]

        # If no models were selected (all weights zero), use the intercept only
        if len(selected_models) == 0:
            logger.warning("No models were selected by Lasso (all weights zero). Using equal weights instead.")
            selected_models = base_models_names
            selected_weights = np.ones(len(selected_models)) / len(selected_models)

        # Normalize weights to sum to 1 as required by VotingRegressor
        if np.sum(selected_weights) != 0:
            normalized_weights = selected_weights / np.sum(np.abs(selected_weights))
        else:
            normalized_weights = np.ones(len(selected_models)) / len(selected_models)

        # Log the selected models and their weights
        logger.info(f"Optimized ensemble selected {len(selected_models)} out of {len(base_models)} models:")
        for model_name, weight, norm_weight in zip(selected_models, selected_weights, normalized_weights):
            logger.info(f"  {model_name}: weight = {weight:.4f}, normalized = {norm_weight:.4f}")

        # Create estimator list for VotingRegressor
        estimators = [(name, base_models[name]) for name in selected_models]

        # Create VotingRegressor with selected models and weights
        ensemble = VotingRegressor(
            estimators=estimators,
            weights=normalized_weights.tolist(),
            n_jobs=-1
        )

        # Fit the ensemble
        ensemble.fit(X, y)

        return ensemble
    def _evaluate_model(self, model, X, y):
        """Evaluate a model and return R2 score."""
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        return r2, y_pred

    def get_best_cv_model(self):
        """
        Return the name of the best model based on cross-validation results.

        Returns:
        --------
        str
            Name of the best model based on cross-validation test R²
        """
        if not hasattr(self, 'cv_results') or not self.cv_results:
            logger.warning("No cross-validation results available")
            return None

        # Find model with best test R²
        best_model = max(self.cv_results.items(), key=lambda x: x[1]['test_r2_mean'])
        return best_model[0]



class ModelEvaluator:
    """Class for evaluating model performance with gap-optimized cross-validation."""

    def __init__(self, experiment_dir, preprocessor=None, feature_selector=None):
        self.experiment_dir = experiment_dir
        self.preprocessor = preprocessor
        self.feature_selector = feature_selector
        self.results_dir = os.path.join(experiment_dir, 'evaluation')
        os.makedirs(self.results_dir, exist_ok=True)

        # Replace EnhancedCrossValidator with GapOptimizedCrossValidator
        self.cross_validator = GapOptimizedCrossValidator(
            n_splits=5,
            n_repeats=3,
            random_state=42,
            experiment_dir=experiment_dir
        )

    def evaluate_models(self, models, X_train, X_test, y_train, y_test, use_cv=True):
        """
        Evaluate all models on train and test sets, and with gap-optimized cross-validation if requested.
        """
        results = {}
        cv_results = {}

        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name} model...")

            # Train predictions
            y_train_pred = model.predict(X_train)

            # Test predictions
            y_test_pred = model.predict(X_test)

            # Calculate metrics
            metrics = self._calculate_metrics(y_train, y_train_pred, y_test, y_test_pred)
            results[model_name] = metrics

            # Generate prediction plots
            self.generate_cv_prediction_plots(X_test, y_test_pred, model, model_name)

            # Generate residual plots
            self._generate_residual_plots(y_test, y_test_pred, model_name)

            # Perform gap-optimized cross-validation if requested
            if use_cv:
                logger.info(f"Performing gap-optimized cross-validation for {model_name}...")

                # Combine train and test for more robust CV
                X_combined = pd.concat([X_train, X_test])
                y_combined = pd.concat([y_train, y_test])

                # Get model-specific CV parameters
                cv_params = get_optimal_cv_params(model_name)

                # Update cross-validator with model-specific parameters
                self.cross_validator.gap_threshold = cv_params['gap_threshold']

                # Perform cross-validation
                cv_result = self.cross_validator.evaluate_model(model, X_combined, y_combined, model_name)
                cv_results[model_name] = cv_result

                # Add CV metrics to regular metrics
                metrics.update({
                    'cv_train_r2_mean': cv_result['train_r2_mean'],
                    'cv_train_r2_std': cv_result['train_r2_std'],
                    'cv_test_r2_mean': cv_result['test_r2_mean'],
                    'cv_test_r2_std': cv_result['test_r2_std'],
                    'cv_r2_gap': cv_result['r2_gap'],
                    'cv_used_folds': cv_result['used_folds'],
                    'cv_total_folds': cv_result['total_folds']
                })

            logger.info(
                f"Model {model_name} - Test R²: {metrics['test_r2']:.4f}, Test RMSE: {metrics['test_rmse']:.4f}")

            if use_cv:
                logger.info(
                    f"Model {model_name} - CV Test R²: {metrics['cv_test_r2_mean']:.4f} +/- {metrics['cv_test_r2_std']:.4f}, Gap: {metrics['cv_r2_gap']:.4f}")

        # Save results to CSV
        results_df = self._format_results(results)
        results_df.to_csv(os.path.join(self.results_dir, 'model_performance.csv'))

        # Store CV results
        self.cv_results = cv_results

        # If cross-validation was performed, create summary
        if cv_results:
            self._create_cv_summary(cv_results)

        return results


    def _calculate_metrics(self, y_train, y_train_pred, y_test, y_test_pred):
        """Calculate evaluation metrics for a model."""
        metrics = {
            'train_r2': r2_score(y_train, y_train_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'test_mae': mean_absolute_error(y_test, y_test_pred)
        }
        return metrics

    def _format_results(self, results):
        """Format results dictionary into a DataFrame."""
        rows = []

        for model_name, metrics in results.items():
            row = {'model': model_name}
            row.update(metrics)
            rows.append(row)

        return pd.DataFrame(rows)

    def _create_cv_summary(self, cv_results):
        """Create summary visualization of cross-validation results."""
        cv_dir = os.path.join(self.results_dir, 'cross_validation')
        os.makedirs(cv_dir, exist_ok=True)

        # Extract key metrics for comparison
        models = list(cv_results.keys())
        train_r2_means = [cv_results[model]['train_r2_mean'] for model in models]
        train_r2_stds = [cv_results[model]['train_r2_std'] for model in models]
        test_r2_means = [cv_results[model]['test_r2_mean'] for model in models]
        test_r2_stds = [cv_results[model]['test_r2_std'] for model in models]
        gaps = [cv_results[model]['r2_gap'] for model in models]

        # Create DataFrame for saving
        summary_df = pd.DataFrame({
            'Model': models,
            'CV_Train_R2_Mean': train_r2_means,
            'CV_Train_R2_Std': train_r2_stds,
            'CV_Test_R2_Mean': test_r2_means,
            'CV_Test_R2_Std': test_r2_stds,
            'CV_R2_Gap': gaps
        })

        # Sort by test R² performance
        summary_df = summary_df.sort_values('CV_Test_R2_Mean', ascending=False)

        # Save to CSV
        summary_df.to_csv(os.path.join(cv_dir, 'cv_summary.csv'), index=False)

        # Create comparison chart
        plt.figure(figsize=(14, 8))

        # Get sorted models and metrics
        sorted_models = summary_df['Model']
        sorted_train_means = summary_df['CV_Train_R2_Mean']
        sorted_test_means = summary_df['CV_Test_R2_Mean']
        sorted_gaps = summary_df['CV_R2_Gap']

        # Set up bar positions
        x = np.arange(len(sorted_models))
        width = 0.25

        # Create bars
        plt.bar(x - width, sorted_train_means, width, label='CV Train R²', color='blue', alpha=0.7)
        plt.bar(x, sorted_test_means, width, label='CV Test R²', color='green', alpha=0.7)
        plt.bar(x + width, sorted_gaps, width, label='Overfitting Gap', color='red', alpha=0.7)

        # Add error bars for train and test R²
        plt.errorbar(x - width, sorted_train_means,
                     yerr=summary_df['CV_Train_R2_Std'],
                     fmt='none', capsize=5, color='black', alpha=0.5)
        plt.errorbar(x, sorted_test_means,
                     yerr=summary_df['CV_Test_R2_Std'],
                     fmt='none', capsize=5, color='black', alpha=0.5)

        # Add labels
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Cross-Validation Results Comparison')
        plt.xticks(x, sorted_models, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add values on bars
        for i, v in enumerate(sorted_train_means):
            plt.text(i - width, v + 0.01, f"{v:.3f}", ha='center', fontsize=9)
        for i, v in enumerate(sorted_test_means):
            plt.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=9)
        for i, v in enumerate(sorted_gaps):
            plt.text(i + width, v + 0.01, f"{v:.3f}", ha='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(cv_dir, 'cv_model_comparison.png'))
        plt.close()

        logger.info(f"Cross-validation summary saved to {cv_dir}")

    def get_best_cv_model(self, cv_results):
        """
        Return the name of the best model based on cross-validation results.

        Parameters:
        -----------
        cv_results : dict
            Cross-validation results dictionary

        Returns:
        --------
        str
            Name of the best model based on cross-validation test R²
        """
        if not cv_results:
            logger.warning("No cross-validation results available")
            return None

        # Find model with best test R²
        best_model = max(cv_results.items(), key=lambda x: x[1]['test_r2_mean'])
        return best_model[0]

    def generate_cv_prediction_plots(self, X, y, model, model_name):
        """
        Generate leak-free cross-validated prediction plots.

        Parameters:
        -----------
        X : DataFrame
            Feature matrix
        y : Series
            Target values
        model : estimator
            Trained model
        model_name : str
            Name of the model for plot title

        Returns:
        --------
        tuple
            (r2, rmse, mae) metrics
        """


        logger.info(f"Generating leak-free cross-validated prediction plot for {model_name}...")

        # Ensure X is a DataFrame (if it's a Series, convert it to a DataFrame)
        if isinstance(X, pd.Series):
            X = pd.DataFrame(X).T  # Convert to a single-row DataFrame

        # Ensure y is a Series (if it's not, convert it)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        # Create empty arrays to store predictions and true values
        all_predictions = []
        all_true_values = []

        # Set up cross-validation
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        # Track successful folds
        successful_folds = 0

        # For each fold
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
            try:
                # Split data
                X_train_fold = X.iloc[train_idx].copy()
                X_test_fold = X.iloc[test_idx].copy()
                y_train_fold = y.iloc[train_idx].copy()
                y_test_fold = y.iloc[test_idx].copy()

                # Explicitly ensure X_train_fold and X_test_fold are DataFrames
                if not isinstance(X_train_fold, pd.DataFrame):
                    X_train_fold = pd.DataFrame(X_train_fold, columns=X.columns)
                if not isinstance(X_test_fold, pd.DataFrame):
                    X_test_fold = pd.DataFrame(X_test_fold, columns=X.columns)

                # Clone model to avoid data leakage
                fold_model = clone(model)

                # Debug info
                logger.info(
                    f"  Fold {fold_idx + 1}: X_train shape: {X_train_fold.shape}, y_train shape: {y_train_fold.shape}")

                # Retrain the model on this fold's training data
                fold_model.fit(X_train_fold, y_train_fold)

                # Predict on this fold's test data
                fold_predictions = fold_model.predict(X_test_fold)

                # Ensure predictions and true values are 1D arrays
                if hasattr(fold_predictions, 'reshape'):
                    fold_predictions = fold_predictions.reshape(-1)
                if hasattr(y_test_fold, 'values'):
                    y_test_values = y_test_fold.values.reshape(-1)
                else:
                    y_test_values = np.array(y_test_fold).reshape(-1)

                # Store predictions and true values
                all_predictions.extend(fold_predictions)
                all_true_values.extend(y_test_values)

                successful_folds += 1

            except Exception as e:
                logger.error(f"Error in fold {fold_idx}: {str(e)}")
                logger.error(
                    f"X_train_fold type: {type(X_train_fold)}, shape: {X_train_fold.shape if hasattr(X_train_fold, 'shape') else 'unknown'}")
                logger.error(
                    f"y_train_fold type: {type(y_train_fold)}, shape: {y_train_fold.shape if hasattr(y_train_fold, 'shape') else 'unknown'}")
                logger.error(f"Skipping this fold")

        # Check if we have predictions
        if len(all_predictions) == 0:
            logger.error(f"No valid predictions were generated across any fold for {model_name}")
            return 0.0, 0.0, 0.0

        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_true_values = np.array(all_true_values)

        # Calculate metrics
        r2 = r2_score(all_true_values, all_predictions)
        rmse = np.sqrt(mean_squared_error(all_true_values, all_predictions))
        mae = mean_absolute_error(all_true_values, all_predictions)

        logger.info(
            f"Leak-free CV metrics for {model_name} (from {successful_folds} folds): R² = {r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}")

        # Create scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(all_true_values, all_predictions, alpha=0.5)

        # Add perfect prediction line
        max_val = max(np.max(all_true_values), np.max(all_predictions))
        min_val = min(np.min(all_true_values), np.min(all_predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')

        # Calculate regression line manually
        #A = np.vstack([all_true_values, np.ones(len(all_true_values))]).T
        #slope, intercept = np.linalg.lstsq(A, all_predictions, rcond=None)[0]

        # Plot regression line
        #x_line = np.linspace(min_val, max_val, 100)
        #y_line = slope * x_line + intercept
        #plt.plot(x_line, y_line, 'g-', alpha=0.7,
                 #label=f'Regression Line (slope={slope:.2f}, intercept={intercept:.2f})')

        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{model_name} - Leak-Free Cross-Validated Predictions')

        # Add metrics to plot
        plt.annotate(f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}\nFolds: {successful_folds}/5',
                     xy=(0.05, 0.95), xycoords='axes fraction', va='top')

        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'{model_name}_leak_free_predictions.png'))
        plt.close()

        # Create residuals plot
        residuals = all_true_values - all_predictions

        plt.figure(figsize=(10, 6))
        plt.scatter(all_predictions, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')

        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title(f'{model_name} - Leak-Free CV Residuals Plot')

        plt.annotate(f'RMSE = {rmse:.4f}\nMAE = {mae:.4f}',
                     xy=(0.05, 0.95), xycoords='axes fraction')

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'{model_name}_leak_free_residuals.png'))
        plt.close()

        return r2, rmse, mae

    def _generate_residual_plots(self, y_true, y_pred, model_name):
        """Generate residual plots."""
        residuals = y_true - y_pred

        plt.figure(figsize=(10, 8))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')

        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title(f'{model_name} - Residual Plot')

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'{model_name}_residuals.png'))
        plt.close()

        # Residual histogram
        plt.figure(figsize=(10, 8))
        plt.hist(residuals, bins=30, alpha=0.7)

        plt.xlabel('Residual Value')
        plt.ylabel('Frequency')
        plt.title(f'{model_name} - Residual Distribution')

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'{model_name}_residual_hist.png'))
        plt.close()

    def generate_shap_analysis(self, model, X, model_name):
        """Generate SHAP explanations for model predictions."""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP package not available, skipping model interpretability")
            return

        try:
            if hasattr(model, 'predict_proba'):
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.Explainer(model)

            # Calculate SHAP values
            shap_values = explainer(X)

            # Create and save SHAP summary plot
            plt.figure(figsize=(12, 10))
            shap.summary_plot(shap_values, X, show=False)
            plt.title(f'{model_name} - SHAP Feature Importance',fontsize=20)
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f'{model_name}_shap_summary.png'))
            plt.close()

            logger.info(f"Generated SHAP analysis for {model_name}")
        except Exception as e:
            logger.error(f"Error generating SHAP analysis for {model_name}: {str(e)}")

    def create_performance_report(self, results):
        """Create a comprehensive performance report."""

        # FIX: Handle the structure returned by integrate_weighted_selector_in_main
        if isinstance(results, dict) and 'weighted_selector' in results:
            # This is the dictionary returned by integrate_weighted_selector_in_main
            logger.info("Extracting results from weighted selector integration...")

            # Extract the comprehensive results
            if 'comprehensive_results' in results:
                model_results = results['comprehensive_results']
            else:
                # Fallback: try to get results from the weighted_selector object
                weighted_selector = results['weighted_selector']
                if hasattr(weighted_selector, 'evaluation_results'):
                    model_results = weighted_selector.evaluation_results
                else:
                    logger.error("Could not extract model results from weighted selector")
                    return

        elif hasattr(results, 'evaluation_results'):
            # This is a WeightedMultiCriteriaModelSelector object directly
            logger.info("Using WeightedMultiCriteriaModelSelector object...")
            model_results = results.evaluation_results

        elif isinstance(results, dict):
            # This is a regular results dictionary
            logger.info("Using regular results dictionary...")
            model_results = results

        else:
            logger.error(f"Unknown results type: {type(results)}")
            logger.error(
                "Expected dict with 'weighted_selector' key, WeightedMultiCriteriaModelSelector object, or regular dict")
            return

        # Filter out models with errors
        valid_results = {k: v for k, v in model_results.items() if 'error' not in v}

        if not valid_results:
            logger.error("No valid model results found for performance report")
            return

        # Create DataFrame for visualization
        models = list(valid_results.keys())

        # Prepare data for bar charts - handle missing keys gracefully
        train_r2 = []
        test_r2 = []
        train_rmse = []
        test_rmse = []

        for model in models:
            result = valid_results[model]
            train_r2.append(result.get('train_r2', 0.0))
            test_r2.append(result.get('test_r2', 0.0))
            train_rmse.append(result.get('train_rmse', 999.0))
            test_rmse.append(result.get('test_rmse', 999.0))

        # Create R² comparison chart
        plt.figure(figsize=(12, 8))
        x = np.arange(len(models))
        width = 0.35

        plt.bar(x - width / 2, train_r2, width, label='Train R²', alpha=0.7)
        plt.bar(x + width / 2, test_r2, width, label='Test R²', alpha=0.7)

        plt.xlabel('Models')
        plt.ylabel('R² Score')
        plt.title('Model Performance Comparison - R² Score')
        plt.xticks(x, models, rotation=45)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Add values on bars
        for i, v in enumerate(train_r2):
            plt.text(i - width / 2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

        for i, v in enumerate(test_r2):
            plt.text(i + width / 2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'model_r2_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Create RMSE comparison chart
        plt.figure(figsize=(12, 8))

        plt.bar(x - width / 2, train_rmse, width, label='Train RMSE', alpha=0.7)
        plt.bar(x + width / 2, test_rmse, width, label='Test RMSE', alpha=0.7)

        plt.xlabel('Models')
        plt.ylabel('RMSE')
        plt.title('Model Performance Comparison - RMSE')
        plt.xticks(x, models, rotation=45)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Add values on bars
        for i, v in enumerate(train_rmse):
            plt.text(i - width / 2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

        for i, v in enumerate(test_rmse):
            plt.text(i + width / 2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'model_rmse_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Add cross-validation results to report if available
        has_cv_results = any('cv_test_r2_mean' in valid_results[model] for model in models)

        if has_cv_results:
            logger.info("Creating cross-validation comparison chart...")

            # Prepare CV data
            cv_test_r2 = []
            cv_test_r2_std = []

            for model in models:
                result = valid_results[model]
                if 'cv_test_r2_mean' in result and not np.isnan(result['cv_test_r2_mean']):
                    cv_test_r2.append(result['cv_test_r2_mean'])
                    cv_test_r2_std.append(result.get('cv_test_r2_std', 0.0))
                else:
                    cv_test_r2.append(None)
                    cv_test_r2_std.append(None)

            # Create CV vs. single-split comparison chart
            plt.figure(figsize=(12, 8))

            # Single-split bars
            bars1 = plt.bar(x - width / 2, test_r2, width, label='Single-Split Test R²', alpha=0.7)

            # Only include bars for models with CV results
            valid_cv_indices = [i for i, r in enumerate(cv_test_r2) if r is not None]

            if valid_cv_indices:
                valid_cv_r2 = [cv_test_r2[i] for i in valid_cv_indices]
                valid_cv_r2_std = [cv_test_r2_std[i] for i in valid_cv_indices]

                # Add CV R² bars with error bars
                bars2 = plt.bar([x[i] + width / 2 for i in valid_cv_indices], valid_cv_r2, width,
                                label='CV Test R²', color='green', alpha=0.7)

                # Add error bars
                plt.errorbar([x[i] + width / 2 for i in valid_cv_indices], valid_cv_r2,
                             yerr=valid_cv_r2_std, fmt='none', capsize=5, color='black', alpha=0.5)

                # Add values on CV bars
                for i, idx in enumerate(valid_cv_indices):
                    plt.text(idx + width / 2, valid_cv_r2[i] + valid_cv_r2_std[i] + 0.01,
                             f'{valid_cv_r2[i]:.3f}±{valid_cv_r2_std[i]:.3f}', ha='center', va='bottom', fontsize=8)

            plt.xlabel('Models')
            plt.ylabel('R² Score')
            plt.title('Single-Split vs. Cross-Validation Performance')
            plt.xticks(x, models, rotation=45)
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            # Add values on single-split bars
            for i, v in enumerate(test_r2):
                plt.text(i - width / 2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'cv_vs_single_split_comparison.png'), dpi=300,
                        bbox_inches='tight')
            plt.close()

        # Create additional weighted criteria visualization if this came from weighted selector
        if isinstance(results, dict) and 'weighted_selector' in results:
            logger.info("Creating weighted criteria visualization...")

            try:
                # Extract prediction accuracy data
                prediction_accuracy = [valid_results[model].get('prediction_accuracy', 0.0) for model in models]

                # Create weighted criteria comparison
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))

                # Plot 1: Prediction Accuracy (50% weight)
                ax1 = axes[0]
                bars1 = ax1.bar(models, prediction_accuracy, color='gold', alpha=0.8)
                ax1.set_ylabel('Prediction Accuracy')
                ax1.set_title('Prediction Accuracy (Weight: 50%)\nHighest Priority')
                ax1.set_xticklabels(models, rotation=45)
                ax1.grid(True, alpha=0.3)

                # Add values on bars
                for bar, value in zip(bars1, prediction_accuracy):
                    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                             f'{value:.3f}', ha='center', va='bottom', fontsize=9)

                # Plot 2: R² Score (35% weight)
                ax2 = axes[1]
                bars2 = ax2.bar(models, test_r2, color='skyblue', alpha=0.8)
                ax2.set_ylabel('Test R² Score')
                ax2.set_title('Model Performance (Weight: 35%)\nSecond Priority')
                ax2.set_xticklabels(models, rotation=45)
                ax2.grid(True, alpha=0.3)

                # Add values on bars
                for bar, value in zip(bars2, test_r2):
                    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                             f'{value:.3f}', ha='center', va='bottom', fontsize=9)

                # Plot 3: RMSE (15% weight)
                ax3 = axes[2]
                bars3 = ax3.bar(models, test_rmse, color='lightcoral', alpha=0.8)
                ax3.set_ylabel('Test RMSE')
                ax3.set_title('RMSE (Weight: 15%)\nLowest Priority')
                ax3.set_xticklabels(models, rotation=45)
                ax3.grid(True, alpha=0.3)

                # Add values on bars
                for bar, value in zip(bars3, test_rmse):
                    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                             f'{value:.3f}', ha='center', va='bottom', fontsize=9)

                plt.tight_layout()
                plt.savefig(os.path.join(self.results_dir, 'weighted_criteria_breakdown.png'), dpi=300,
                            bbox_inches='tight')
                plt.close()

            except Exception as e:
                logger.warning(f"Could not create weighted criteria visualization: {str(e)}")

        logger.info("Performance report created successfully")
        logger.info(f"Charts saved to: {self.results_dir}")

    def get_best_model_name(self, results):
        """Return the name of the best performing model based on results type."""

        # Handle weighted selector results
        if isinstance(results, dict) and 'best_model_name' in results:
            return results['best_model_name']
        elif isinstance(results, dict) and 'weighted_selector' in results:
            weighted_selector = results['weighted_selector']
            return weighted_selector.get_best_model_name()
        elif hasattr(results, 'get_best_model_name'):
            return results.get_best_model_name()
        elif isinstance(results, dict):
            # Fallback: find model with highest test_r2
            best_model = max(results.items(), key=lambda x: x[1].get('test_r2', -999) if 'error' not in x[1] else -999)
            return best_model[0]
        else:
            logger.warning(f"Unknown results type for getting best model: {type(results)}")
            return None