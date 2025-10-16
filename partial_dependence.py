import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
from sklearn.utils.validation import check_is_fitted
import os
import logging

logger = logging.getLogger(__name__)


class PartialDependenceAnalyzer:
    """Class for creating and analyzing partial dependence plots."""

    def __init__(self, experiment_dir):
        self.experiment_dir = experiment_dir
        self.pdp_dir = os.path.join(experiment_dir, 'partial_dependence')
        os.makedirs(self.pdp_dir, exist_ok=True)

    def _check_model_fitted(self, model):
        """
        Check if the model is fitted and can be used for partial dependence analysis.

        Parameters:
        -----------
        model : trained model object
            Model to check

        Returns:
        --------
        bool : True if model is fitted and ready for PDP analysis
        """
        try:
            # Try sklearn's check_is_fitted
            check_is_fitted(model)
            return True
        except:
            # For custom models, check if they have prediction capability
            try:
                if hasattr(model, 'predict') and callable(getattr(model, 'predict')):
                    # Additional checks for custom BayesianModelAveraging
                    if hasattr(model, 'models_') and model.models_:
                        return True
                    elif hasattr(model, 'is_fitted') and model.is_fitted:
                        return True
                    elif hasattr(model, 'fitted_') and model.fitted_:
                        return True
                    else:
                        logger.warning("Model may not be properly fitted. Attempting to use anyway.")
                        return True
                return False
            except:
                return False

    def _manual_partial_dependence(self, model, X, feature_idx, n_grid_points=50):
        """
        Manually calculate partial dependence when sklearn's function fails.

        Parameters:
        -----------
        model : trained model object
            Model to analyze
        X : DataFrame
            Feature matrix
        feature_idx : int
            Index of the feature to analyze
        n_grid_points : int
            Number of points to evaluate along the feature range

        Returns:
        --------
        dict : Dictionary with 'grid_values' and 'average' keys
        """
        feature_name = X.columns[feature_idx]
        feature_values = X.iloc[:, feature_idx]

        # Create grid of values for the feature, handling potential issues
        try:
            # Handle both numeric and categorical features
            if pd.api.types.is_numeric_dtype(feature_values):
                # For numeric features, create a linear grid
                min_val = feature_values.min()
                max_val = feature_values.max()
                if min_val == max_val:
                    # Handle constant features
                    grid_values = np.array([min_val])
                else:
                    grid_values = np.linspace(min_val, max_val, n_grid_points)
            else:
                # For categorical features, use unique values
                grid_values = feature_values.unique()
                if len(grid_values) > n_grid_points:
                    # Sample if too many categories
                    grid_values = np.random.choice(grid_values, n_grid_points, replace=False)
        except Exception as e:
            logger.error(f"Error creating grid for feature {feature_name}: {str(e)}")
            return {'grid_values': [np.array([])], 'average': [np.array([])]}

        # Calculate partial dependence manually
        pdp_values = []

        # Use a smaller sample if dataset is very large to speed up calculation
        if len(X) > 1000:
            sample_size = min(1000, len(X))
            X_sample = X.sample(sample_size, random_state=42)
            logger.info(f"Using sample of {sample_size} rows for PDP calculation")
        else:
            X_sample = X

        for grid_val in grid_values:
            try:
                # Create modified dataset with feature set to grid_val
                X_modified = X_sample.copy()
                X_modified.iloc[:, feature_idx] = grid_val

                # Get predictions for all samples
                predictions = model.predict(X_modified)

                # Handle different prediction formats
                if hasattr(predictions, 'flatten'):
                    predictions = predictions.flatten()
                elif isinstance(predictions, list):
                    predictions = np.array(predictions)

                # Calculate mean prediction
                mean_pred = np.mean(predictions)
                pdp_values.append(mean_pred)

            except Exception as e:
                logger.warning(f"Error predicting for grid value {grid_val} in feature {feature_name}: {str(e)}")
                # Use the mean of previous values or 0 if no previous values
                if pdp_values:
                    pdp_values.append(np.mean(pdp_values))
                else:
                    pdp_values.append(0.0)

        # Convert to numpy arrays
        grid_values = np.array(grid_values)
        pdp_values = np.array(pdp_values)

        # Remove any NaN or infinite values
        valid_mask = np.isfinite(pdp_values)
        if not np.any(valid_mask):
            logger.error(f"All PDP values are invalid for feature {feature_name}")
            return {'grid_values': [np.array([])], 'average': [np.array([])]}

        grid_values = grid_values[valid_mask]
        pdp_values = pdp_values[valid_mask]

        logger.info(f"Manual PDP calculation completed for {feature_name}: {len(grid_values)} points")

        return {
            'grid_values': [grid_values],
            'average': [pdp_values]
        }

    def generate_single_feature_pdp(self, model, X, feature_names=None, n_features=None):
        """
        Generate partial dependence plots for top features.

        Parameters:
        -----------
        model : trained model object
            Model to analyze
        X : DataFrame
            Feature matrix
        feature_names : list, optional
            Specific features to analyze. If None, will use all or top n_features
        n_features : int, optional
            Number of top features to analyze if feature_names is None
        """
        # Check if model is fitted
        if not self._check_model_fitted(model):
            logger.error("Model is not fitted. Please fit the model before generating PDPs.")
            return

        if feature_names is None:
            if hasattr(model, 'feature_importances_'):
                # Use feature importance to select top features
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                if n_features is None:
                    n_features = min(10, len(X.columns))
                feature_names = [X.columns[i] for i in indices[:n_features]]
            else:
                # No feature importance available, use all features or top n
                if n_features is None:
                    feature_names = X.columns.tolist()
                else:
                    feature_names = X.columns.tolist()[:n_features]

        # Ensure feature_names is a list
        if isinstance(feature_names, str):
            feature_names = [feature_names]

        logger.info(f"Generating partial dependence plots for features: {feature_names}")

        # Generate individual PDP for each feature
        for feature in feature_names:
            fig, ax = plt.subplots(figsize=(10, 6))

            try:
                # First check if this is a sklearn-compatible model
                from sklearn.utils.validation import check_is_fitted
                check_is_fitted(model)
                # Try sklearn's partial_dependence
                pdp = partial_dependence(model, X, features=[X.columns.get_loc(feature)], kind='average')
                feature_values = pdp['grid_values'][0]
                pdp_values = pdp['average'][0]
                logger.debug(f"Successfully used sklearn partial_dependence for {feature}")
            except Exception as e:
                logger.info(
                    f"sklearn partial_dependence not compatible with model for {feature}: {type(e).__name__}. Using manual calculation.")
                # Fall back to manual calculation
                try:
                    pdp = self._manual_partial_dependence(model, X, X.columns.get_loc(feature))
                    feature_values = pdp['grid_values'][0]
                    pdp_values = pdp['average'][0]
                    logger.debug(f"Successfully used manual partial_dependence for {feature}")
                except Exception as e2:
                    logger.error(f"Manual partial dependence calculation failed for {feature}: {str(e2)}")
                    plt.close()
                    continue

            # Plot the partial dependence
            ax.plot(feature_values, pdp_values, 'b-', linewidth=2)

            # Add distribution of feature values
            ax_twin = ax.twinx()
            sns.kdeplot(X[feature], ax=ax_twin, color='r', alpha=0.3)
            ax_twin.set_ylabel('Data Density', color='r')
            ax_twin.tick_params(axis='y', colors='r')

            # Set labels and title
            ax.set_xlabel(f'Feature: {feature}',fontsize=14)
            ax.set_ylabel('Partial Dependence',fontsize=14)
            ax.set_title(f'Partial Dependence Plot for {feature}',fontsize=16)
            ax.grid(True, linestyle='--', alpha=0.7)

            # Save the plot
            file_path = os.path.join(self.pdp_dir, f'pdp_{feature.replace(" ", "_")}.png')
            plt.tight_layout()
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()

        logger.info(f"Individual PDPs saved to {self.pdp_dir}")

    def generate_feature_grid_pdp(self, model, X, feature_names=None, n_features=6):
        """
        Generate a grid of partial dependence plots for top features.

        Parameters:
        -----------
        model : trained model object
            Model to analyze
        X : DataFrame
            Feature matrix
        feature_names : list, optional
            Specific features to analyze. If None, will use top n_features
        n_features : int
            Number of top features to include in the grid
        """
        # Check if model is fitted
        if not self._check_model_fitted(model):
            logger.error("Model is not fitted. Please fit the model before generating PDPs.")
            return

        if feature_names is None:
            if hasattr(model, 'feature_importances_'):
                # Use feature importance to select top features
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                feature_names = [X.columns[i] for i in indices[:n_features]]
            else:
                # No feature importance available, use top n features
                feature_names = X.columns.tolist()[:n_features]

        logger.info(f"Generating grid of partial dependence plots for features: {feature_names}")

        fig, axes = plt.subplots(nrows=(n_features + 1) // 2, ncols=2, figsize=(16, 12))

        # Flatten axes array for easier indexing
        axes = axes.flatten()

        for i, feature in enumerate(feature_names):
            if i < len(axes):
                try:
                    # Try sklearn's PartialDependenceDisplay first
                    PartialDependenceDisplay.from_estimator(
                        model, X, [X.columns.get_loc(feature)],
                        kind='average', ax=axes[i], subsample=1000
                    )
                except Exception as e:
                    logger.warning(
                        f"sklearn PartialDependenceDisplay failed for {feature}: {str(e)}. Using manual calculation.")
                    # Fall back to manual calculation and plotting
                    try:
                        pdp = self._manual_partial_dependence(model, X, X.columns.get_loc(feature))
                        feature_values = pdp['grid_values'][0]
                        pdp_values = pdp['average'][0]

                        axes[i].plot(feature_values, pdp_values, 'b-', linewidth=2)
                        axes[i].set_xlabel(feature)
                        axes[i].set_ylabel('Partial Dependence')
                    except Exception as e2:
                        logger.error(f"Manual PDP calculation failed for {feature}: {str(e2)}")
                        axes[i].text(0.5, 0.5, f'Error generating PDP for {feature}',
                                     ha='center', va='center', transform=axes[i].transAxes)

                # Customize plot
                axes[i].set_title(f'PDP for {feature}',fontsize=16)
                axes[i].grid(True, linestyle='--', alpha=0.7)

        # Hide any unused axes
        for i in range(len(feature_names), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        file_path = os.path.join(self.pdp_dir, 'pdp_feature_grid.png')
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Grid PDP saved to {file_path}")

    def generate_2d_interaction_pdp(self, model, X, feature_pairs=None, top_n=3, total_pb_feature='TotalPb',
                                    max_pairs=5, timeout=300, grid_resolution=20, subsample=200):
        """
        Generate 2D partial dependence plots with optimizations to prevent freezing.

        Parameters:
        -----------
        model : trained model object
            Model to analyze
        X : DataFrame
            Feature matrix
        feature_pairs : list of tuples, optional
            Specific feature pairs to analyze. If None, will analyze interactions
            between TotalPb and other top features
        top_n : int, default=3
            Number of top features to consider for interactions (reduced from 5)
        total_pb_feature : str
            Name of the total lead feature in the dataset
        max_pairs : int, default=5
            Maximum number of feature pairs to analyze
        timeout : int, default=300
            Maximum seconds to allow for each pair (5 minutes)
        grid_resolution : int, default=20
            Resolution of the grid for partial dependence calculation (reduced from default)
        subsample : int, default=200
            Number of samples to use for calculation (reduced from 1000)
        """
        # Check if model is fitted
        if not self._check_model_fitted(model):
            logger.error("Model is not fitted. Please fit the model before generating PDPs.")
            return

        import signal
        from contextlib import contextmanager
        import time

        @contextmanager
        def time_limit(seconds):
            """Context manager to limit execution time."""

            def signal_handler(signum, frame):
                raise TimeoutError(f"Timed out after {seconds} seconds")

            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(seconds)
            try:
                yield
            finally:
                signal.alarm(0)

        if feature_pairs is None:
            # Generate interactions between TotalPb and other important features
            if total_pb_feature not in X.columns:
                logger.warning(f"TotalPb feature '{total_pb_feature}' not found in dataset")
                if hasattr(model, 'feature_importances_'):
                    # Use feature importance to select top features
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    top_features = [X.columns[i] for i in indices[:top_n + 1]]
                    # Create pairs of top features
                    feature_pairs = [(top_features[0], feature) for feature in top_features[1:top_n + 1]]
                else:
                    # No feature importance available, use first feature with others
                    top_features = X.columns.tolist()[:top_n + 1]
                    feature_pairs = [(top_features[0], feature) for feature in top_features[1:top_n + 1]]
            else:
                # Use TotalPb with other important features
                if hasattr(model, 'feature_importances_'):
                    # Get features sorted by importance
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    feature_names = [X.columns[i] for i in indices]

                    # Remove TotalPb from this list if present
                    if total_pb_feature in feature_names:
                        feature_names.remove(total_pb_feature)

                    # Create pairs
                    feature_pairs = [(total_pb_feature, feature) for feature in feature_names[:top_n]]
                else:
                    # No feature importance available
                    feature_names = [f for f in X.columns if f != total_pb_feature][:top_n]
                    feature_pairs = [(total_pb_feature, feature) for feature in feature_names]

        # Limit the number of pairs to analyze
        if len(feature_pairs) > max_pairs:
            logger.warning(f"Limiting analysis to {max_pairs} feature pairs to prevent excessive runtime")
            feature_pairs = feature_pairs[:max_pairs]

        logger.info(f"Generating 2D PDPs for feature pairs: {feature_pairs}")

        # Create a custom colormap
        cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#2874A6', '#FFFFFF', '#7D3C98'], N=256)

        for pair in feature_pairs:
            feature1, feature2 = pair

            # Check if features exist in the dataset
            if feature1 not in X.columns or feature2 not in X.columns:
                logger.warning(f"Feature pair {pair} not found in dataset")
                continue

            # Get feature indices
            feature1_idx = X.columns.get_loc(feature1)
            feature2_idx = X.columns.get_loc(feature2)

            try:
                # Use time limit to prevent hanging
                with time_limit(timeout):
                    start_time = time.time()
                    logger.info(f"Starting PDP calculation for {feature1} vs {feature2}")

                    # Sample the dataset if it's large to speed up calculation
                    if len(X) > 1000:
                        sample_size = min(1000, len(X))
                        X_sample = X.sample(sample_size, random_state=42)
                    else:
                        X_sample = X

                    plt.figure(figsize=(12, 10))

                    try:
                        # Try sklearn's PartialDependenceDisplay first
                        PartialDependenceDisplay.from_estimator(
                            model, X_sample, features=[(feature1_idx, feature2_idx)],
                            kind='average',
                            subsample=subsample,  # Reduced from 1000
                            grid_resolution=grid_resolution,  # Add explicit grid resolution
                            contour_kw={'cmap': cmap}
                        )
                    except Exception as e:
                        logger.warning(
                            f"sklearn 2D PDP failed for {feature1} vs {feature2}: {str(e)}. Skipping this pair.")
                        plt.close('all')
                        continue

                    plt.suptitle(f'2D Partial Dependence: {feature1} vs {feature2}', fontsize=16)

                    # Save the plot
                    file_path = os.path.join(self.pdp_dir, f'pdp_2d_{feature1}_{feature2}.png')
                    plt.tight_layout()
                    plt.savefig(file_path, dpi=300, bbox_inches='tight')
                    plt.close('all')  # Close all figures to prevent memory leaks

                    elapsed_time = time.time() - start_time
                    logger.info(f"Completed PDP for {feature1} vs {feature2} in {elapsed_time:.2f} seconds")

            except TimeoutError:
                logger.warning(f"PDP calculation for {feature1} vs {feature2} timed out after {timeout} seconds.")
                plt.close('all')  # Ensure figures are closed
            except Exception as e:
                logger.error(f"Error generating PDP for {feature1} vs {feature2}: {str(e)}")
                plt.close('all')  # Ensure figures are closed

        logger.info(f"2D PDPs saved to {self.pdp_dir}")

    def analyze_totalPb_interactions(self, model, X, y, total_pb_feature='TotalPb', top_n=5):
        """
        Analyze how TotalPb interacts with other features by creating conditional PDPs.

        Parameters:
        -----------
        model : trained model object
            Model to analyze
        X : DataFrame
            Feature matrix
        y : Series
            Target variable
        total_pb_feature : str
            Name of the total lead feature
        top_n : int
            Number of top features to analyze interactions with
        """
        # Check if model is fitted
        if not self._check_model_fitted(model):
            logger.error("Model is not fitted. Please fit the model before generating PDPs.")
            return

        if total_pb_feature not in X.columns:
            logger.warning(f"TotalPb feature '{total_pb_feature}' not found in dataset")
            return

        # Get other important features
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            feature_names = [X.columns[i] for i in indices if X.columns[i] != total_pb_feature][:top_n]
        else:
            feature_names = [f for f in X.columns if f != total_pb_feature][:top_n]

        logger.info(f"Analyzing interactions between {total_pb_feature} and: {feature_names}")

        # Create percentile bins for TotalPb
        totalPb_values = X[total_pb_feature]
        low_threshold = totalPb_values.quantile(0.25)
        high_threshold = totalPb_values.quantile(0.75)

        # Create masks for different percentile bins
        low_mask = totalPb_values <= low_threshold
        med_mask = (totalPb_values > low_threshold) & (totalPb_values < high_threshold)
        high_mask = totalPb_values >= high_threshold

        masks = {
            'Low TotalPb (<=25%)': low_mask,
            'Mid TotalPb (25-75%)': med_mask,
            'High TotalPb (>=75%)': high_mask
        }

        # Generate conditional PDPs for each feature
        for feature in feature_names:
            plt.figure(figsize=(12, 8))

            for label, mask in masks.items():
                if mask.sum() >= 20:  # Ensure enough samples in each bin
                    # Subset data
                    X_subset = X[mask]

                    try:
                        # Try sklearn's partial_dependence first
                        pdp = partial_dependence(model, X_subset, features=[X.columns.get_loc(feature)], kind='average')
                        feature_values = pdp['grid_values'][0]
                        pdp_values = pdp['average'][0]
                    except Exception as e:
                        logger.warning(
                            f"sklearn partial_dependence failed for {feature} with mask {label}: {str(e)}. Using manual calculation.")
                        # Fall back to manual calculation
                        try:
                            pdp = self._manual_partial_dependence(model, X_subset, X.columns.get_loc(feature))
                            feature_values = pdp['grid_values'][0]
                            pdp_values = pdp['average'][0]
                        except Exception as e2:
                            logger.error(
                                f"Manual partial dependence calculation failed for {feature} with mask {label}: {str(e2)}")
                            continue

                    # Plot
                    plt.plot(feature_values, pdp_values, linewidth=2, label=label)

            plt.xlabel(feature)
            plt.ylabel('Partial Dependence')
            plt.title(f'Conditional Partial Dependence of {feature} at Different TotalPb Levels')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)

            # Save plot
            file_path = os.path.join(self.pdp_dir, f'conditional_pdp_{feature}_by_TotalPb.png')
            plt.tight_layout()
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()

        logger.info(f"Conditional PDPs saved to {self.pdp_dir}")

    def generate_ice_plots(self, model, X, feature_names=None, n_samples=50, n_features=5):
        """
        Generate Individual Conditional Expectation (ICE) plots for top features.

        Parameters:
        -----------
        model : trained model object
            Model to analyze
        X : DataFrame
            Feature matrix
        feature_names : list, optional
            Specific features to analyze. If None, will use top n_features
        n_samples : int
            Number of individual samples to plot
        n_features : int
            Number of top features to analyze if feature_names is None
        """
        # Check if model is fitted
        if not self._check_model_fitted(model):
            logger.error("Model is not fitted. Please fit the model before generating PDPs.")
            return

        if feature_names is None:
            if hasattr(model, 'feature_importances_'):
                # Use feature importance to select top features
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                feature_names = [X.columns[i] for i in indices[:n_features]]
            else:
                # No feature importance available, use top n features
                feature_names = X.columns.tolist()[:n_features]

        logger.info(f"Generating ICE plots for features: {feature_names}")

        # Sample rows from X for individual lines (to avoid overcrowding)
        if len(X) > n_samples:
            sample_indices = np.random.choice(len(X), n_samples, replace=False)
            X_sampled = X.iloc[sample_indices]
        else:
            X_sampled = X

        for feature in feature_names:
            try:
                plt.figure(figsize=(12, 8))

                # Get feature index
                feature_idx = X.columns.get_loc(feature)

                try:
                    # Try sklearn's ICE plot first
                    display = PartialDependenceDisplay.from_estimator(
                        model, X_sampled, features=[feature_idx],
                        kind='individual', centered=True,
                        ax=plt.gca(), subsample=n_samples
                    )

                    # Overlay averaged PD curve
                    # The correct way to add the average line:
                    # Create a new partial dependence display for the average
                    pd_avg = PartialDependenceDisplay.from_estimator(
                        model, X_sampled, features=[feature_idx],
                        kind='average',
                        ax=plt.gca(),
                        line_kw={'color': 'black', 'linewidth': 3, 'label': 'Average (PDP)'}
                    )
                except Exception as e:
                    logger.warning(
                        f"sklearn ICE plot failed for {feature}: {str(e)}. ICE plots require sklearn compatibility.")
                    # For ICE plots, we can't easily create a manual fallback
                    # So we'll skip this feature and log the issue
                    plt.close('all')
                    continue

                plt.title(f'Individual Conditional Expectation for {feature}')
                plt.xlabel(feature)
                plt.ylabel('Centered Prediction')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()

                # Save plot
                file_path = os.path.join(self.pdp_dir, f'ice_{feature}.png')
                plt.tight_layout()
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                plt.close('all')

            except Exception as e:
                logger.error(f"Error generating ICE plot for feature {feature}: {str(e)}")
                plt.close('all')  # Ensure figures are closed even on error

        logger.info(f"ICE plots saved to {self.pdp_dir}")