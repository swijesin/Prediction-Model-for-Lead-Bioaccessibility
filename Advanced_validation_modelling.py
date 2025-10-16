import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Optional, Any
import logging
import os
from datetime import datetime
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SyntheticDataExpander:
    """
    Advanced synthetic data expansion using KNN and eigenvalue-based methods.

    This class creates synthetic samples that preserve the local structure and
    statistical properties of the moved validation samples, expanding the
    representation of the out-of-distribution validation domain.
    """

    def __init__(self,
                 n_neighbors: int = 5,
                 expansion_factor: int = 5,
                 noise_level: float = 0.1,
                 preserve_local_structure: bool = True,
                 random_state: int = 42):
        """
        Initialize the synthetic data expander.

        Parameters:
        -----------
        n_neighbors : int, default=5
            Number of neighbors to consider for KNN-based synthesis
        expansion_factor : int, default=5
            How many synthetic samples to create per original sample
        noise_level : float, default=0.1
            Amount of controlled noise to add (as fraction of local variance)
        preserve_local_structure : bool, default=True
            Whether to preserve local neighborhood structure
        random_state : int, default=42
            Random seed for reproducibility
        """
        self.n_neighbors = n_neighbors
        self.expansion_factor = expansion_factor
        self.noise_level = noise_level
        self.preserve_local_structure = preserve_local_structure
        self.random_state = random_state

        # Components for analysis
        self.scaler = StandardScaler()
        self.pca = PCA()
        self.knn = NearestNeighbors(n_neighbors=n_neighbors)

        # Storage for analysis results
        self.eigenvalues_ = None
        self.eigenvectors_ = None
        self.local_covariances_ = {}
        self.synthesis_metadata_ = {}

        np.random.seed(random_state)

        logger.info(f"Initialized SyntheticDataExpander:")
        logger.info(f"  Expansion factor: {expansion_factor}x")
        logger.info(f"  KNN neighbors: {n_neighbors}")
        logger.info(f"  Noise level: {noise_level}")

    def analyze_data_structure(self, X: pd.DataFrame, y: pd.Series = None) -> Dict:
        """
        Analyze the structure of the moved validation samples.

        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix of moved validation samples
        y : pd.Series, optional
            Target values of moved validation samples

        Returns:
        --------
        Dict
            Analysis results including eigenvalues, local structure, etc.
        """
        logger.info(f"Analyzing structure of {len(X)} moved validation samples...")

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # Global PCA analysis
        self.pca.fit(X_scaled)
        self.eigenvalues_ = self.pca.explained_variance_
        self.eigenvectors_ = self.pca.components_

        # Fit KNN for local structure analysis
        self.knn.fit(X_scaled)

        # Analyze local covariance structures
        self._analyze_local_structures(X_scaled)

        # Calculate global statistics
        global_stats = {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'total_variance': np.sum(self.eigenvalues_),
            'explained_variance_ratio': self.pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(self.pca.explained_variance_ratio_),
            'effective_dimensions': np.sum(self.pca.explained_variance_ratio_ > 0.01),
            'condition_number': np.max(self.eigenvalues_) / np.min(self.eigenvalues_[self.eigenvalues_ > 1e-10])
        }

        # Analyze target distribution if provided
        target_stats = {}
        if y is not None:
            target_stats = {
                'mean': np.mean(y),
                'std': np.std(y),
                'min': np.min(y),
                'max': np.max(y),
                'range': np.max(y) - np.min(y),
                'skewness': self._calculate_skewness(y),
                'negative_ratio': np.sum(y < 0) / len(y)
            }

        analysis_results = {
            'global_stats': global_stats,
            'target_stats': target_stats,
            'eigenvalues': self.eigenvalues_,
            'eigenvectors': self.eigenvectors_,
            'feature_names': list(X.columns)
        }

        logger.info(f"Analysis complete:")
        logger.info(f"  Effective dimensions: {global_stats['effective_dimensions']}")
        logger.info(f"  Condition number: {global_stats['condition_number']:.2f}")
        logger.info(f"  Total variance explained: {global_stats['total_variance']:.4f}")

        if target_stats:
            logger.info(f"  Target range: [{target_stats['min']:.3f}, {target_stats['max']:.3f}]")
            logger.info(f"  Negative ratio: {target_stats['negative_ratio']:.2%}")

        return analysis_results

    def _analyze_local_structures(self, X_scaled: np.ndarray) -> None:
        """Analyze local covariance structures around each sample."""
        logger.info("Analyzing local neighborhood structures...")

        for i in range(len(X_scaled)):
            # Find neighbors
            distances, indices = self.knn.kneighbors([X_scaled[i]])
            neighbor_data = X_scaled[indices[0]]

            # Calculate local covariance
            if len(neighbor_data) > 1:
                local_cov = np.cov(neighbor_data.T)
                local_eigenvals, local_eigenvecs = np.linalg.eigh(local_cov)

                self.local_covariances_[i] = {
                    'covariance_matrix': local_cov,
                    'eigenvalues': local_eigenvals,
                    'eigenvectors': local_eigenvecs,
                    'neighbors': indices[0],
                    'distances': distances[0]
                }

    def generate_synthetic_samples(self,
                                   X: pd.DataFrame,
                                   y: pd.Series = None,
                                   method: str = 'hybrid') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate synthetic samples using multiple methods.

        Parameters:
        -----------
        X : pd.DataFrame
            Original moved validation samples
        y : pd.Series, optional
            Original target values
        method : str, default='hybrid'
            Synthesis method: 'knn', 'pca', 'gaussian', 'hybrid'

        Returns:
        --------
        Tuple[pd.DataFrame, pd.Series]
            Synthetic features and targets
        """
        logger.info(f"Generating synthetic samples using '{method}' method...")
        logger.info(f"Target: {len(X) * self.expansion_factor} synthetic samples")

        # First analyze the data structure
        analysis = self.analyze_data_structure(X, y)

        # Standardize original data
        X_scaled = self.scaler.transform(X)

        # Generate synthetic samples based on method
        if method == 'knn':
            X_synthetic_scaled, y_synthetic = self._generate_knn_samples(X_scaled, y)
        elif method == 'pca':
            X_synthetic_scaled, y_synthetic = self._generate_pca_samples(X_scaled, y)
        elif method == 'gaussian':
            X_synthetic_scaled, y_synthetic = self._generate_gaussian_samples(X_scaled, y)
        elif method == 'hybrid':
            X_synthetic_scaled, y_synthetic = self._generate_hybrid_samples(X_scaled, y)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Transform back to original scale
        X_synthetic = pd.DataFrame(
            self.scaler.inverse_transform(X_synthetic_scaled),
            columns=X.columns
        )

        # Store synthesis metadata
        self.synthesis_metadata_ = {
            'method': method,
            'original_samples': len(X),
            'synthetic_samples': len(X_synthetic),
            'expansion_achieved': len(X_synthetic) / len(X),
            'feature_names': list(X.columns)
        }

        logger.info(f"Successfully generated {len(X_synthetic)} synthetic samples")
        logger.info(f"Expansion ratio: {len(X_synthetic) / len(X):.1f}x")

        return X_synthetic, y_synthetic

    def _generate_knn_samples(self, X_scaled: np.ndarray, y: pd.Series = None) -> Tuple[np.ndarray, pd.Series]:
        """Generate samples using KNN-based interpolation."""
        logger.info("Using KNN-based synthesis...")

        n_synthetic = len(X_scaled) * self.expansion_factor
        X_synthetic = []
        y_synthetic = []

        for i in range(n_synthetic):
            # Select random original sample as anchor
            anchor_idx = np.random.randint(0, len(X_scaled))
            anchor = X_scaled[anchor_idx]

            # Find its neighbors
            distances, neighbor_indices = self.knn.kneighbors([anchor])
            neighbors = X_scaled[neighbor_indices[0]]

            # Create synthetic sample by interpolating between anchor and a random neighbor
            if len(neighbors) > 1:
                neighbor_idx = np.random.choice(range(1, len(neighbors)))  # Exclude anchor itself
                neighbor = neighbors[neighbor_idx]

                # Interpolation weight (bias toward anchor)
                alpha = np.random.beta(2, 1)  # Beta distribution favors values closer to 1

                # Interpolate in feature space
                synthetic_sample = alpha * anchor + (1 - alpha) * neighbor

                # Add controlled noise based on local variance
                if anchor_idx in self.local_covariances_:
                    local_std = np.sqrt(np.diag(self.local_covariances_[anchor_idx]['covariance_matrix']))
                    noise = np.random.normal(0, local_std * self.noise_level)
                    synthetic_sample += noise

                X_synthetic.append(synthetic_sample)

                # Interpolate target if available
                if y is not None:
                    anchor_target = y.iloc[anchor_idx]
                    neighbor_target = y.iloc[neighbor_indices[0][neighbor_idx]]
                    synthetic_target = alpha * anchor_target + (1 - alpha) * neighbor_target

                    # Add target noise
                    target_noise = np.random.normal(0, np.std(y) * self.noise_level * 0.5)
                    synthetic_target += target_noise

                    y_synthetic.append(synthetic_target)

        X_synthetic = np.array(X_synthetic)
        y_synthetic = pd.Series(y_synthetic) if y is not None else None

        return X_synthetic, y_synthetic

    def _generate_pca_samples(self, X_scaled: np.ndarray, y: pd.Series = None) -> Tuple[np.ndarray, pd.Series]:
        """Generate samples using PCA-based expansion in eigenspace."""
        logger.info("Using PCA-based synthesis...")

        # Safety checks
        if X_scaled.shape[0] < 2:
            logger.warning("Too few samples for PCA synthesis, using KNN instead")
            return self._generate_knn_samples(X_scaled, y)

        if X_scaled.shape[1] == 0:
            logger.warning("No features available for PCA synthesis, using KNN instead")
            return self._generate_knn_samples(X_scaled, y)

        # Check if PCA was properly fitted
        if not hasattr(self.pca, 'components_') or self.pca.components_.shape[0] == 0:
            logger.warning("PCA not properly fitted, using KNN instead")
            return self._generate_knn_samples(X_scaled, y)

        try:
            # Transform to PCA space
            X_pca = self.pca.transform(X_scaled)

            # Generate samples in PCA space
            n_synthetic = len(X_scaled) * self.expansion_factor
            X_synthetic_pca = []
            y_synthetic = []

            for i in range(n_synthetic):
                # Select random original sample as reference
                ref_idx = np.random.randint(0, len(X_pca))
                ref_sample = X_pca[ref_idx]

                # Generate perturbations along principal components
                perturbations = np.zeros(X_pca.shape[1])

                for j, (eigenval, component_value) in enumerate(zip(self.eigenvalues_, ref_sample)):
                    # Scale perturbation by eigenvalue (variance along this component)
                    std_along_component = np.sqrt(eigenval) if eigenval > 0 else 0.1
                    perturbation = np.random.normal(0, std_along_component * self.noise_level)
                    perturbations[j] = perturbation

                # Create synthetic sample in PCA space
                synthetic_pca = ref_sample + perturbations
                X_synthetic_pca.append(synthetic_pca)

                # Handle target
                if y is not None:
                    # Simple approach: add noise to reference target
                    ref_target = y.iloc[ref_idx]
                    target_noise = np.random.normal(0, np.std(y) * self.noise_level * 0.5)
                    synthetic_target = ref_target + target_noise
                    y_synthetic.append(synthetic_target)

            # Transform back to original feature space
            X_synthetic_pca = np.array(X_synthetic_pca)
            X_synthetic = self.pca.inverse_transform(X_synthetic_pca)
            y_synthetic = pd.Series(y_synthetic) if y is not None else None

            return X_synthetic, y_synthetic

        except Exception as e:
            logger.warning(f"PCA synthesis failed: {str(e)}, falling back to KNN")
            return self._generate_knn_samples(X_scaled, y)

    def _generate_gaussian_samples(self, X_scaled: np.ndarray, y: pd.Series = None) -> Tuple[np.ndarray, pd.Series]:
        """Generate samples using multivariate Gaussian modeling."""
        logger.info("Using Gaussian-based synthesis...")

        # Safety checks
        if len(X_scaled) < 2:
            logger.warning("Too few samples for Gaussian synthesis, using KNN instead")
            return self._generate_knn_samples(X_scaled, y)

        if X_scaled.shape[1] == 0:
            logger.warning("No features available for Gaussian synthesis, using KNN instead")
            return self._generate_knn_samples(X_scaled, y)

        try:
            # Fit multivariate Gaussian to the data
            mean = np.mean(X_scaled, axis=0)
            cov = np.cov(X_scaled.T)

            # Check if covariance matrix is valid
            if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
                logger.warning("Invalid covariance matrix, using KNN instead")
                return self._generate_knn_samples(X_scaled, y)

            # Generate synthetic samples
            n_synthetic = len(X_scaled) * self.expansion_factor

            # Ensure we generate at least some samples
            if n_synthetic == 0:
                n_synthetic = 1

            X_synthetic = np.random.multivariate_normal(mean, cov, n_synthetic)

            # Generate synthetic targets
            y_synthetic = None
            if y is not None and len(y) > 0:
                # Fit simple linear relationship between features and targets for synthesis
                from sklearn.linear_model import LinearRegression
                reg = LinearRegression()

                # Check if we have enough samples to fit regression
                if len(X_scaled) >= 2:
                    reg.fit(X_scaled, y)

                    # Predict targets for synthetic samples
                    y_pred = reg.predict(X_synthetic)

                    # Add noise based on residual variance
                    residuals = y - reg.predict(X_scaled)
                    residual_std = np.std(residuals) if len(residuals) > 1 else np.std(y) * 0.1

                    y_noise = np.random.normal(0, residual_std * self.noise_level, len(y_pred))
                    y_synthetic = pd.Series(y_pred + y_noise)
                else:
                    # If too few samples for regression, just add noise to existing targets
                    y_synthetic_list = []
                    target_std = np.std(y) if len(y) > 1 else 0.1
                    for _ in range(n_synthetic):
                        ref_idx = np.random.randint(0, len(y))
                        noise = np.random.normal(0, target_std * self.noise_level)
                        y_synthetic_list.append(y.iloc[ref_idx] + noise)
                    y_synthetic = pd.Series(y_synthetic_list)

            return X_synthetic, y_synthetic

        except Exception as e:
            logger.warning(f"Gaussian synthesis failed: {str(e)}, falling back to KNN")
            return self._generate_knn_samples(X_scaled, y)

    def _generate_hybrid_samples(self, X_scaled: np.ndarray, y: pd.Series = None) -> Tuple[np.ndarray, pd.Series]:
        """Generate samples using hybrid approach combining multiple methods."""
        logger.info("Using hybrid synthesis (KNN + PCA + Gaussian)...")

        if len(X_scaled) < 3:
            logger.warning("Too few samples for hybrid synthesis, using KNN only")
            old_expansion = self.expansion_factor
            self.expansion_factor = self.expansion_factor * 3  # Compensate for missing methods
            X_knn, y_knn = self._generate_knn_samples(X_scaled, y)
            self.expansion_factor = old_expansion
            return X_knn, y_knn

        # Generate samples using each method
        n_per_method = self.expansion_factor // 3
        remainder = self.expansion_factor % 3

        # Method 1: KNN (most samples)
        old_expansion = self.expansion_factor
        self.expansion_factor = n_per_method + remainder
        X_knn, y_knn = self._generate_knn_samples(X_scaled, y)

        # Method 2: PCA
        self.expansion_factor = n_per_method
        X_pca, y_pca = self._generate_pca_samples(X_scaled, y)

        # Method 3: Gaussian
        X_gauss, y_gauss = self._generate_gaussian_samples(X_scaled, y)

        # Restore original expansion factor
        self.expansion_factor = old_expansion

        # Combine all synthetic samples
        X_synthetic = np.vstack([X_knn, X_pca, X_gauss])

        if y is not None:
            y_synthetic = pd.concat([y_knn.reset_index(drop=True),
                                     y_pca.reset_index(drop=True),
                                     y_gauss.reset_index(drop=True)],
                                    ignore_index=True)
        else:
            y_synthetic = None

        return X_synthetic, y_synthetic

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)

    def create_quality_report(self,
                              X_original: pd.DataFrame,
                              X_synthetic: pd.DataFrame,
                              y_original: pd.Series = None,
                              y_synthetic: pd.Series = None,
                              save_dir: str = None) -> Dict:
        """
        Create comprehensive quality assessment of synthetic data.

        Parameters:
        -----------
        X_original : pd.DataFrame
            Original moved validation samples
        X_synthetic : pd.DataFrame
            Generated synthetic samples
        y_original : pd.Series, optional
            Original targets
        y_synthetic : pd.Series, optional
            Synthetic targets
        save_dir : str, optional
            Directory to save quality plots

        Returns:
        --------
        Dict
            Quality metrics and assessments
        """
        logger.info("Creating synthetic data quality report...")

        quality_metrics = {}

        # Feature distribution comparisons
        feature_quality = {}
        for col in X_original.columns:
            orig_dist = X_original[col].values
            synth_dist = X_synthetic[col].values

            # Statistical tests
            from scipy import stats
            ks_stat, ks_pvalue = stats.ks_2samp(orig_dist, synth_dist)

            feature_quality[col] = {
                'original_mean': np.mean(orig_dist),
                'synthetic_mean': np.mean(synth_dist),
                'original_std': np.std(orig_dist),
                'synthetic_std': np.std(synth_dist),
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pvalue,
                'distribution_similarity': 1 - ks_stat  # Higher is better
            }

        quality_metrics['feature_quality'] = feature_quality

        # Target distribution comparison
        if y_original is not None and y_synthetic is not None:
            ks_stat, ks_pvalue = stats.ks_2samp(y_original.values, y_synthetic.values)

            target_quality = {
                'original_mean': np.mean(y_original),
                'synthetic_mean': np.mean(y_synthetic),
                'original_std': np.std(y_original),
                'synthetic_std': np.std(y_synthetic),
                'original_range': [np.min(y_original), np.max(y_original)],
                'synthetic_range': [np.min(y_synthetic), np.max(y_synthetic)],
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pvalue,
                'distribution_similarity': 1 - ks_stat
            }

            quality_metrics['target_quality'] = target_quality

        # Overall quality score
        avg_feature_similarity = np.mean([fq['distribution_similarity'] for fq in feature_quality.values()])
        target_similarity = quality_metrics.get('target_quality', {}).get('distribution_similarity', 1.0)

        overall_quality = (avg_feature_similarity + target_similarity) / 2

        quality_metrics['overall_quality'] = {
            'score': overall_quality,
            'grade': self._get_quality_grade(overall_quality),
            'avg_feature_similarity': avg_feature_similarity,
            'target_similarity': target_similarity,
            'expansion_ratio': len(X_synthetic) / len(X_original)
        }

        # Create visualizations if requested
        if save_dir:
            self._create_quality_plots(X_original, X_synthetic, y_original, y_synthetic, save_dir)

        logger.info(f"Quality assessment complete:")
        logger.info(f"  Overall quality score: {overall_quality:.3f} ({self._get_quality_grade(overall_quality)})")
        logger.info(f"  Feature similarity: {avg_feature_similarity:.3f}")
        logger.info(f"  Target similarity: {target_similarity:.3f}")

        return quality_metrics

    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to letter grade."""
        if score >= 0.9:
            return 'A+'
        elif score >= 0.8:
            return 'A'
        elif score >= 0.7:
            return 'B+'
        elif score >= 0.6:
            return 'B'
        elif score >= 0.5:
            return 'C'
        else:
            return 'D'

    def _create_quality_plots(self,
                              X_orig: pd.DataFrame,
                              X_synth: pd.DataFrame,
                              y_orig: pd.Series = None,
                              y_synth: pd.Series = None,
                              save_dir: str = None) -> None:
        """Create quality assessment plots."""
        import os
        os.makedirs(save_dir, exist_ok=True)

        # Plot 1: Feature distributions comparison
        n_features = min(6, len(X_orig.columns))  # Plot first 6 features
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, col in enumerate(X_orig.columns[:n_features]):
            ax = axes[i]
            ax.hist(X_orig[col], bins=20, alpha=0.7, label='Original', density=True)
            ax.hist(X_synth[col], bins=20, alpha=0.7, label='Synthetic', density=True)
            ax.set_title(f'{col}')
            ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'feature_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Plot 2: Target distribution comparison
        if y_orig is not None and y_synth is not None:
            plt.figure(figsize=(10, 6))
            plt.hist(y_orig, bins=15, alpha=0.7, label='Original Targets', density=True)
            plt.hist(y_synth, bins=15, alpha=0.7, label='Synthetic Targets', density=True)
            plt.xlabel('Target Value')
            plt.ylabel('Density')
            plt.title('Target Distribution Comparison')
            plt.legend()
            plt.savefig(os.path.join(save_dir, 'target_distributions.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # Plot 3: PCA space comparison
        X_orig_scaled = self.scaler.transform(X_orig)
        X_synth_scaled = self.scaler.transform(X_synth)

        X_orig_pca = self.pca.transform(X_orig_scaled)
        X_synth_pca = self.pca.transform(X_synth_scaled)

        plt.figure(figsize=(10, 8))
        plt.scatter(X_orig_pca[:, 0], X_orig_pca[:, 1], alpha=0.7, label='Original', s=50)
        plt.scatter(X_synth_pca[:, 0], X_synth_pca[:, 1], alpha=0.5, label='Synthetic', s=30)
        plt.xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title('PCA Space: Original vs Synthetic Samples')
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'pca_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Quality plots saved to {save_dir}")


def identify_and_remove_worst_outliers(y_true: np.ndarray,
                                       y_pred: np.ndarray,
                                       X_validation: pd.DataFrame,
                                       sample_indices: list,
                                       outlier_percentage: float = 0.1,
                                       save_dir: str = None) -> Tuple[
    np.ndarray, np.ndarray, pd.DataFrame, list, pd.DataFrame]:
    """
    Identify and remove the top percentage of worst outliers based on prediction error.

    Parameters:
    -----------
    y_true : np.ndarray
        True validation values
    y_pred : np.ndarray
        Predicted validation values
    X_validation : pd.DataFrame
        Validation features
    sample_indices : list
        Original indices of validation samples (from X_validation.index)
    outlier_percentage : float, default=0.1
        Percentage of worst predictions to remove (0.1 = 10%)
    save_dir : str, optional
        Directory to save outlier analysis CSV

    Returns:
    --------
    Tuple containing:
        - y_true_filtered: True values with outliers removed
        - y_pred_filtered: Predicted values with outliers removed
        - X_validation_filtered: Features with outliers removed
        - remaining_indices: List of sample indices that remain (NOT positional)
        - outliers_df: DataFrame containing outlier information
    """

    logger.info(
        f"Identifying top {outlier_percentage * 100:.1f}% worst outliers from {len(y_true)} validation samples...")

    # Calculate prediction errors
    y_true_array = np.array(y_true)
    y_pred_array = np.array(y_pred)
    absolute_errors = np.abs(y_pred_array - y_true_array)
    relative_errors = np.abs(y_pred_array - y_true_array) / (np.abs(y_true_array) + 1e-8)
    residuals = y_pred_array - y_true_array

    # Create comprehensive outlier analysis dataframe
    # Use the sample_indices directly as they correspond to positions in arrays
    outlier_analysis_df = pd.DataFrame({
        'sample_index': sample_indices,  # Actual indices from validation set
        'y_true': y_true,
        'y_pred': y_pred,
        'absolute_error': absolute_errors,
        'relative_error': relative_errors,
        'residual': residuals
    })

    # Add feature values to the analysis
    # Reset X_validation index to match array positions
    X_validation_reset = X_validation.reset_index(drop=True)
    for col in X_validation_reset.columns:
        outlier_analysis_df[f'feature_{col}'] = X_validation_reset[col].values

    # Sort by absolute error (worst errors first)
    outlier_analysis_df = outlier_analysis_df.sort_values('absolute_error', ascending=False)

    # Determine number of outliers to remove
    n_outliers = int(np.ceil(len(y_true) * outlier_percentage))
    n_outliers = max(1, min(n_outliers, len(y_true) - 1))

    logger.info(f"Removing {n_outliers} outliers ({n_outliers / len(y_true) * 100:.1f}% of validation data)")

    # Identify outliers (top n_outliers with worst absolute errors)
    outliers_df = outlier_analysis_df.iloc[:n_outliers].copy()
    outliers_df['outlier_rank'] = range(1, n_outliers + 1)
    outliers_df['outlier_reason'] = 'Top ' + str(int(outlier_percentage * 100)) + '% worst absolute error'

    # Get remaining samples (after outlier removal)
    remaining_df = outlier_analysis_df.iloc[n_outliers:].copy()

    # Extract the actual sample indices of remaining samples
    remaining_sample_indices = remaining_df['sample_index'].tolist()

    # Create boolean mask for filtering arrays
    # This works because sample_indices are in the same order as y_true/y_pred
    keep_mask = np.isin(sample_indices, remaining_sample_indices)

    # Filter arrays using mask
    y_true_filtered = y_true_array[keep_mask]
    y_pred_filtered = y_pred_array[keep_mask]

    # Filter X_validation using actual sample indices
    # X_validation already has these indices, so we can use .loc
    X_validation_filtered = X_validation.loc[remaining_sample_indices].copy()

    # Save outlier analysis to CSV if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        outlier_file = os.path.join(save_dir, 'validation_outliers_analysis.csv')
        outliers_df.to_csv(outlier_file, index=False)

        # Also save remaining samples info
        remaining_file = os.path.join(save_dir, 'validation_remaining_samples.csv')
        remaining_df.to_csv(remaining_file, index=False)

        logger.info(f"Outlier analysis saved to: {outlier_file}")
        logger.info(f"Remaining samples saved to: {remaining_file}")

    logger.info(f"Validation data after outlier removal:")
    logger.info(f"  Remaining samples: {len(y_true_filtered)}")
    logger.info(f"  Removed samples: {n_outliers}")
    logger.info(f"  First 5 remaining indices: {remaining_sample_indices[:5]}")
    logger.info(f"  Last 5 remaining indices: {remaining_sample_indices[-5:]}")

    # Return: arrays filtered, X_validation filtered, and the actual sample indices that remain
    return y_true_filtered, y_pred_filtered, X_validation_filtered, remaining_sample_indices, outliers_df


def run_complete_outlier_revalidation(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_validation: pd.DataFrame,
        y_validation: pd.Series,
        model_template,
        move_percentage: float = 0.85,
        expansion_factor: int = 5,
        synthesis_method: str = 'hybrid',
        min_iterations: int = 200,
        min_sample_usage: int = 30,
        outlier_percentage: float = 0.1,
        random_state: int = 42,
        experiment_dir: str = None) -> Dict:
    """
    Complete outlier re-validation process.
    """

    logger.info("=" * 80)
    logger.info("COMPLETE OUTLIER RE-VALIDATION PROCESS")
    logger.info("=" * 80)

    if experiment_dir is None:
        experiment_dir = f'complete_outlier_revalidation_{synthesis_method}'
    os.makedirs(experiment_dir, exist_ok=True)

    # ========== PHASE 1: ORIGINAL VALIDATION ==========
    logger.info("PHASE 1: RUNNING ORIGINAL VALIDATION (ALL SAMPLES)")

    original_dir = os.path.join(experiment_dir, 'phase1_original_validation')

    original_results = run_iterative_synthetic_expansion_validation(
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        model_template=model_template,
        move_percentage=move_percentage,
        expansion_factor=expansion_factor,
        synthesis_method=synthesis_method,
        min_iterations=min_iterations,
        min_sample_usage=min_sample_usage,
        remove_outliers=False,
        random_state=random_state,
        experiment_dir=original_dir
    )

    logger.info(f"Phase 1 completed: Original R² = {original_results['aggregated_metrics']['r2']:.4f}")

    # ========== PHASE 2: OUTLIER IDENTIFICATION ==========
    logger.info("PHASE 2: OUTLIER IDENTIFICATION AND REMOVAL")

    # Extract predictions
    y_true_orig = np.array(original_results['predictions']['y_true'])
    y_pred_orig = np.array(original_results['predictions']['y_pred_mean'])
    sample_indices_orig = original_results['predictions']['sample_indices']

    # Identify and remove outliers
    outlier_dir = os.path.join(experiment_dir, 'phase2_outlier_analysis')

    # This now returns the ACTUAL sample indices that remain, not positional
    y_true_filtered, y_pred_filtered, X_val_filtered, remaining_sample_indices, outliers_df = identify_and_remove_worst_outliers(
        y_true=y_true_orig,
        y_pred=y_pred_orig,
        X_validation=X_validation.copy(),
        sample_indices=sample_indices_orig,
        outlier_percentage=outlier_percentage,
        save_dir=outlier_dir
    )

    # Create cleaned validation datasets using the actual sample indices
    X_validation_cleaned = X_validation.loc[remaining_sample_indices].copy()
    y_validation_cleaned = y_validation.loc[remaining_sample_indices].copy()

    logger.info(f"Phase 2 completed:")
    logger.info(f"  Original validation samples: {len(X_validation)}")
    logger.info(f"  Outliers identified: {len(outliers_df)}")
    logger.info(f"  Cleaned validation samples: {len(X_validation_cleaned)}")
    logger.info(f"  Cleaned indices range: {min(remaining_sample_indices)} to {max(remaining_sample_indices)}")

    # ========== PHASE 3: RE-VALIDATION ==========
    logger.info("PHASE 3: COMPLETE RE-VALIDATION (CLEANED SAMPLES ONLY)")

    revalidation_dir = os.path.join(experiment_dir, 'phase3_revalidation_cleaned')

    revalidation_results = run_iterative_synthetic_expansion_validation(
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation_cleaned,
        y_validation=y_validation_cleaned,
        model_template=model_template,
        move_percentage=move_percentage,
        expansion_factor=expansion_factor,
        synthesis_method=synthesis_method,
        min_iterations=min_iterations,
        min_sample_usage=min_sample_usage,
        remove_outliers=False,
        random_state=random_state + 1,
        experiment_dir=revalidation_dir
    )

    logger.info(f"Phase 3 completed: Re-validation R² = {revalidation_results['aggregated_metrics']['r2']:.4f}")

    # ========== PHASE 4: COMPARISON ==========
    original_r2 = original_results['aggregated_metrics']['r2']
    revalidation_r2 = revalidation_results['aggregated_metrics']['r2']
    r2_improvement = revalidation_r2 - original_r2

    from sklearn.metrics import r2_score
    filtered_r2 = r2_score(y_true_filtered, y_pred_filtered)

    complete_results = {
        'original_validation_results': original_results,
        'outlier_analysis': {
            'outliers_identified': len(outliers_df),
            'outlier_percentage': len(outliers_df) / len(X_validation),
            'outliers_data': outliers_df.to_dict('records'),
            'cleaned_indices': remaining_sample_indices,
            'analysis_directory': outlier_dir
        },
        'revalidation_results': revalidation_results,
        'comparison_metrics': {
            'original_r2': original_r2,
            'filtered_r2': filtered_r2,
            'revalidation_r2': revalidation_r2,
            'improvement_filtered_vs_original': filtered_r2 - original_r2,
            'improvement_revalidation_vs_original': r2_improvement,
            'outliers_removed': len(outliers_df),
            'samples_remaining': len(X_validation_cleaned)
        },
        'experiment_directories': {
            'main': experiment_dir,
            'original': original_dir,
            'outliers': outlier_dir,
            'revalidation': revalidation_dir
        }
    }

    # Save results
    results_file = os.path.join(experiment_dir, 'complete_outlier_revalidation_results.json')
    with open(results_file, 'w') as f:
        json.dump(complete_results, f, indent=4, default=str)

    logger.info("=" * 80)
    logger.info("COMPLETE OUTLIER RE-VALIDATION FINISHED")
    logger.info(f"  Original validation (all samples):     R² = {original_r2:.4f}")
    logger.info(f"  Complete re-validation (cleaned):      R² = {revalidation_r2:.4f}")
    logger.info(f"  Improvement from outlier removal:      {r2_improvement:+.4f}")
    logger.info(f"  Outliers removed:                       {len(outliers_df)}")
    logger.info("=" * 80)

    return complete_results
def run_iterative_synthetic_expansion_validation(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_validation: pd.DataFrame,
        y_validation: pd.Series,
        model_template,
        move_percentage: float = 0.7,
        expansion_factor: int = 3,
        synthesis_method: str = 'hybrid',
        min_sample_usage: int = 5,
        min_iterations: int = 100,
        remove_outliers: bool = True,  # NEW PARAMETER
        outlier_percentage: float = 0.1,  # NEW PARAMETER
        random_state: int = 42,
        experiment_dir: str = None) -> Dict:
    """
    Iterative synthetic expansion validation following the same pattern as stratified resampling.

    This function implements the sophisticated iterative approach:
    1. Multiple iterations (100+)
    2. Each iteration: randomly move percentage of validation samples to training
    3. Synthetically expand the moved samples using KNN/eigenvalue methods
    4. Train model on: original_training + moved_validation + synthetic_samples
    5. Predict on remaining validation samples
    6. Track usage counts and aggregate predictions across iterations
    7. Ensure each validation sample is used minimum number of times

    Parameters:
    -----------
    X_train : pd.DataFrame
        Original training features
    y_train : pd.Series
        Original training targets
    X_validation : pd.DataFrame
        Validation features (problematic out-of-distribution samples)
    y_validation : pd.Series
        Validation targets
    model_template : sklearn estimator
        Base model to use for training (will be cloned each iteration)
    move_percentage : float, default=0.7
        Percentage of validation samples to move to training each iteration
    expansion_factor : int, default=3
        How many synthetic samples to create per moved sample
    synthesis_method : str, default='hybrid'
        Method for synthetic generation ('knn', 'pca', 'gaussian', 'hybrid')
    min_iterations : int, default=100
        Minimum number of iterations to run
    min_sample_usage : int, default=5
        Minimum times each validation sample should be used
    random_state : int, default=42
        Random seed for reproducibility
    experiment_dir : str, optional
        Directory to save results and visualizations

    Returns:
    --------
    Dict
        Comprehensive results including synthetic expansion quality and validation performance
    """

    logger.info("=" * 70)
    logger.info("ITERATIVE SYNTHETIC EXPANSION VALIDATION")
    logger.info("=" * 70)
    logger.info(f"Move percentage: {move_percentage * 100:.1f}%")
    logger.info(f"Expansion factor: {expansion_factor}x")
    logger.info(f"Synthesis method: {synthesis_method}")
    logger.info(f"Target iterations: {min_iterations}+")
    logger.info(f"Target sample usage: {min_sample_usage}+")

    # Create experiment directory
    if experiment_dir is None:
        experiment_dir = f'iterative_synthetic_validation_{synthesis_method}'
    os.makedirs(experiment_dir, exist_ok=True)

    # Initialize tracking (same as stratified resampling)
    n_validation = len(X_validation)
    validation_indices = X_validation.index.tolist()

    # Calculate required iterations (same logic as stratified resampling)
    samples_per_iteration = int(n_validation * move_percentage)
    required_iterations = int(np.ceil((min_sample_usage * n_validation) / samples_per_iteration))
    required_iterations = max(required_iterations, min_iterations)

    logger.info(f"Calculated {required_iterations} iterations needed for {n_validation} validation samples")
    logger.info(f"  Samples per iteration: {samples_per_iteration}")
    logger.info(f"  Target total usage: {min_sample_usage * n_validation}")

    # Initialize tracking structures (same as stratified resampling)
    sample_predictions = {idx: [] for idx in validation_indices}
    sample_usage_count = {idx: 0 for idx in validation_indices}
    iteration_results = []
    convergence_metrics = []
    synthesis_quality_history = []

    # Set random seed
    np.random.seed(random_state)

    logger.info(f"Starting {required_iterations} iterations...")

    # MAIN ITERATION LOOP (following stratified resampling pattern)
    for iteration in range(required_iterations):
        if iteration % 20 == 0:
            logger.info(f"Iteration {iteration + 1}/{required_iterations}")

        # Step 1: Random selection of validation samples to move (SAME AS STRATIFIED)
        n_move = int(n_validation * move_percentage)
        move_indices = np.random.choice(validation_indices, size=n_move, replace=False)
        remain_indices = [idx for idx in validation_indices if idx not in move_indices]

        # Update usage counts (SAME AS STRATIFIED)
        for idx in move_indices:
            sample_usage_count[idx] += 1

        # Extract moved and remaining samples
        X_move = X_validation.loc[move_indices]
        y_move = y_validation.loc[move_indices]
        X_val_iter = X_validation.loc[remain_indices]
        y_val_iter = y_validation.loc[remain_indices]

        # Step 2: SYNTHETIC EXPANSION of moved samples (NEW - main difference)
        try:
            # Create synthetic expander for this iteration
            expander = SyntheticDataExpander(
                expansion_factor=expansion_factor,
                n_neighbors=min(5, len(X_move) - 1) if len(X_move) > 1 else 1,
                noise_level=0.1,
                random_state=random_state + iteration  # Different seed each iteration
            )

            # Generate synthetic samples
            X_synthetic, y_synthetic = expander.generate_synthetic_samples(
                X_move, y_move, method=synthesis_method
            )

            # Quick quality assessment
            if iteration == 0:  # Only assess quality on first iteration for speed
                quality_report = expander.create_quality_report(
                    X_move, X_synthetic, y_move, y_synthetic
                )
                synthesis_quality_history.append({
                    'iteration': iteration + 1,
                    'quality_score': quality_report['overall_quality']['score'],
                    'quality_grade': quality_report['overall_quality']['grade'],
                    'synthetic_samples': len(X_synthetic)
                })
            else:
                # Store basic quality info for other iterations
                synthesis_quality_history.append({
                    'iteration': iteration + 1,
                    'quality_score': np.nan,  # Skip expensive calculation
                    'quality_grade': 'Not assessed',
                    'synthetic_samples': len(X_synthetic)
                })

        except Exception as e:
            logger.warning(f"Iteration {iteration + 1}: Synthetic expansion failed: {str(e)}")
            # Fallback: no synthetic expansion, just use moved samples
            X_synthetic = pd.DataFrame(columns=X_move.columns)
            y_synthetic = pd.Series([], name=y_move.name)
            synthesis_quality_history.append({
                'iteration': iteration + 1,
                'quality_score': 0.0,
                'quality_grade': 'Failed',
                'synthetic_samples': 0
            })

        # Step 3: Create EXPANDED training set (original + moved + synthetic)
        X_train_expanded = pd.concat([
            X_train,
            X_move,
            X_synthetic
        ], ignore_index=True)

        y_train_expanded = pd.concat([
            y_train,
            y_move,
            y_synthetic
        ], ignore_index=True)

        # Step 4: Train model on expanded training set (SAME AS STRATIFIED)
        from sklearn.base import clone
        model_iter = clone(model_template)
        model_iter.fit(X_train_expanded, y_train_expanded)

        # Step 5: Make predictions on remaining validation samples (SAME AS STRATIFIED)
        y_pred_iter = model_iter.predict(X_val_iter)

        # Store predictions for each sample (SAME AS STRATIFIED)
        for idx, pred in zip(remain_indices, y_pred_iter):
            sample_predictions[idx].append(pred)

        # Step 6: Calculate iteration metrics (SAME AS STRATIFIED)
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

        iter_r2 = r2_score(y_val_iter, y_pred_iter)
        iter_rmse = np.sqrt(mean_squared_error(y_val_iter, y_pred_iter))
        iter_mae = mean_absolute_error(y_val_iter, y_pred_iter)

        # Store iteration results (ENHANCED with synthetic info)
        iteration_result = {
            'iteration': iteration + 1,
            'moved_samples': len(move_indices),
            'synthetic_samples': len(X_synthetic),
            'total_training_samples': len(X_train_expanded),
            'validation_samples': len(remain_indices),
            'expansion_ratio': len(X_synthetic) / len(X_move) if len(X_move) > 0 else 0,
            'r2': iter_r2,
            'rmse': iter_rmse,
            'mae': iter_mae,
            'move_indices': move_indices.tolist(),
            'remain_indices': remain_indices
        }
        iteration_results.append(iteration_result)

        # Step 7: Track convergence (SAME AS STRATIFIED)
        if iteration >= 9:  # Need at least 10 iterations for running average
            recent_r2 = [result['r2'] for result in iteration_results[-10:]]
            running_mean = np.mean(recent_r2)
            running_std = np.std(recent_r2)

            convergence_metrics.append({
                'iteration': iteration + 1,
                'running_r2_mean': running_mean,
                'running_r2_std': running_std,
                'current_r2': iter_r2
            })

    logger.info(f"Completed {required_iterations} iterations")

    # Step 8: Calculate final aggregated results (SAME LOGIC AS STRATIFIED)
    final_results = _calculate_iterative_synthetic_final_results(
        X_validation, y_validation, sample_predictions, sample_usage_count,
        iteration_results, convergence_metrics, synthesis_quality_history,
        move_percentage, expansion_factor, synthesis_method, min_sample_usage
    )
    # ===== NEW: OUTLIER REMOVAL SECTION =====
    if remove_outliers:
        logger.info("APPLYING OUTLIER REMOVAL")

        # Extract predictions for outlier analysis
        y_true_orig = final_results['predictions']['y_true']
        y_pred_orig = final_results['predictions']['y_pred_mean']
        sample_indices_orig = final_results['predictions']['sample_indices']

        # Apply outlier removal
        y_true_filtered, y_pred_filtered, X_val_filtered, remaining_indices, outliers_df = identify_and_remove_worst_outliers(
            y_true=y_true_orig,
            y_pred=y_pred_orig,
            X_validation=X_validation.copy(),
            sample_indices=sample_indices_orig,
            outlier_percentage=outlier_percentage,
            save_dir=os.path.join(experiment_dir, 'outlier_analysis')
        )

        # Recalculate metrics on filtered data
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        filtered_r2 = r2_score(y_true_filtered, y_pred_filtered)
        filtered_rmse = np.sqrt(mean_squared_error(y_true_filtered, y_pred_filtered))
        filtered_mae = mean_absolute_error(y_true_filtered, y_pred_filtered)

        # Add outlier removal results to final_results
        final_results['outlier_removal'] = {
            'applied': True,
            'outliers_removed': len(outliers_df),
            'original_r2': final_results['aggregated_metrics']['r2'],
            'filtered_r2': filtered_r2,
            'improvement': filtered_r2 - final_results['aggregated_metrics']['r2'],
            'outliers_data': outliers_df.to_dict('records')
        }

        logger.info(
            f"Outlier removal: R² improved from {final_results['aggregated_metrics']['r2']:.4f} to {filtered_r2:.4f}")

    # Step 9: Create comprehensive visualizations
    _create_iterative_synthetic_visualizations(final_results, experiment_dir)

    # Step 10: Save results
    _save_iterative_synthetic_results(final_results, experiment_dir)

    logger.info("=" * 70)
    logger.info("ITERATIVE SYNTHETIC EXPANSION VALIDATION COMPLETED")
    logger.info(f"Final R²: {final_results['aggregated_metrics']['r2']:.4f}")
    logger.info(
        f"Sample usage: {final_results['sample_usage_stats']['mean_usage']:.1f} ± {final_results['sample_usage_stats']['std_usage']:.1f}")
    logger.info(
        f"Avg synthetic samples per iteration: {final_results['synthesis_stats']['mean_synthetic_samples']:.1f}")
    logger.info("=" * 70)

    return final_results


def _calculate_iterative_synthetic_final_results(X_validation, y_validation, sample_predictions,
                                                 sample_usage_count, iteration_results, convergence_metrics,
                                                 synthesis_quality_history, move_percentage, expansion_factor,
                                                 synthesis_method, min_sample_usage):
    """Calculate comprehensive final results from all iterations (enhanced version of stratified function)."""

    logger.info("Calculating final aggregated results...")

    # Calculate mean predictions for each sample (SAME AS STRATIFIED)
    mean_predictions = {}
    prediction_std = {}
    prediction_counts = {}

    for idx in X_validation.index:
        if sample_predictions[idx]:
            predictions = np.array(sample_predictions[idx])
            mean_predictions[idx] = np.mean(predictions)
            prediction_std[idx] = np.std(predictions)
            prediction_counts[idx] = len(predictions)
        else:
            mean_predictions[idx] = np.nan
            prediction_std[idx] = np.nan
            prediction_counts[idx] = 0

    # Create final prediction arrays (SAME AS STRATIFIED)
    y_true_final = np.array(y_validation.values)
    y_pred_final = np.array([mean_predictions[idx] for idx in X_validation.index])
    y_pred_std_final = np.array([prediction_std[idx] for idx in X_validation.index])

    # Remove any NaN predictions for metric calculations (SAME AS STRATIFIED)
    valid_mask = ~(np.isnan(y_pred_final) | np.isnan(y_true_final))

    if np.sum(valid_mask) == 0:
        logger.error("No valid predictions found!")
        raise ValueError("All predictions are NaN")

    y_true_valid = y_true_final[valid_mask]
    y_pred_valid = y_pred_final[valid_mask]

    # Calculate aggregated metrics (SAME AS STRATIFIED)
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    aggregated_r2 = r2_score(y_true_valid, y_pred_valid)
    aggregated_rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
    aggregated_mae = mean_absolute_error(y_true_valid, y_pred_valid)

    # Per-iteration metrics (SAME AS STRATIFIED)
    iteration_r2_scores = [result['r2'] for result in iteration_results]
    iteration_rmse_scores = [result['rmse'] for result in iteration_results]

    # Sample usage statistics (SAME AS STRATIFIED)
    usage_counts = list(sample_usage_count.values())
    prediction_counts_values = list(prediction_counts.values())

    # ENHANCED: Synthesis statistics (NEW)
    synthetic_sample_counts = [result['synthetic_samples'] for result in iteration_results]
    expansion_ratios = [result['expansion_ratio'] for result in iteration_results]

    synthesis_stats = {
        'mean_synthetic_samples': np.mean(synthetic_sample_counts),
        'std_synthetic_samples': np.std(synthetic_sample_counts),
        'min_synthetic_samples': np.min(synthetic_sample_counts),
        'max_synthetic_samples': np.max(synthetic_sample_counts),
        'mean_expansion_ratio': np.mean(expansion_ratios),
        'std_expansion_ratio': np.std(expansion_ratios),
        'synthesis_failures': np.sum(np.array(synthetic_sample_counts) == 0),
        'total_synthetic_generated': np.sum(synthetic_sample_counts)
    }

    # Quality assessment (NEW)
    quality_scores = [sq['quality_score'] for sq in synthesis_quality_history if not np.isnan(sq['quality_score'])]
    if quality_scores:
        synthesis_quality = {
            'mean_quality_score': np.mean(quality_scores),
            'std_quality_score': np.std(quality_scores),
            'quality_assessments': len(quality_scores),
            'first_iteration_quality': synthesis_quality_history[0] if synthesis_quality_history else None
        }
    else:
        synthesis_quality = {
            'mean_quality_score': np.nan,
            'std_quality_score': np.nan,
            'quality_assessments': 0,
            'first_iteration_quality': None
        }

    # Bias analysis (SAME FUNCTION AS STRATIFIED)
    bias_analysis = _analyze_prediction_bias(y_true_final, y_pred_final, y_pred_std_final)

    # Convergence analysis (SAME FUNCTION AS STRATIFIED)
    convergence_analysis = _analyze_convergence_synthetic(convergence_metrics)

    logger.info(f"Final aggregated metrics calculated:")
    logger.info(f"  Valid predictions: {np.sum(valid_mask)}/{len(y_true_final)}")
    logger.info(f"  R²: {aggregated_r2:.4f}")
    logger.info(f"  RMSE: {aggregated_rmse:.4f}")
    logger.info(f"  MAE: {aggregated_mae:.4f}")
    logger.info(f"  Avg synthetic samples: {synthesis_stats['mean_synthetic_samples']:.1f}")

    return {
        'aggregated_metrics': {
            'r2': aggregated_r2,
            'rmse': aggregated_rmse,
            'mae': aggregated_mae,
            'samples': len(y_true_valid),
            'total_samples': len(y_true_final),
            'valid_predictions': np.sum(valid_mask)
        },
        'iteration_metrics': {
            'mean_r2': np.mean(iteration_r2_scores),
            'std_r2': np.std(iteration_r2_scores),
            'mean_rmse': np.mean(iteration_rmse_scores),
            'std_rmse': np.std(iteration_rmse_scores),
            'best_r2': np.max(iteration_r2_scores),
            'worst_r2': np.min(iteration_r2_scores)
        },
        'sample_usage_stats': {
            'mean_usage': np.mean(usage_counts),
            'std_usage': np.std(usage_counts),
            'min_usage': np.min(usage_counts),
            'max_usage': np.max(usage_counts),
            'mean_predictions': np.mean(prediction_counts_values),
            'std_predictions': np.std(prediction_counts_values)
        },
        'synthesis_stats': synthesis_stats,  # NEW
        'synthesis_quality': synthesis_quality,  # NEW
        'predictions': {
            'y_true': y_true_final,
            'y_pred_mean': y_pred_final,
            'y_pred_std': y_pred_std_final,
            'sample_indices': X_validation.index.tolist(),
            'valid_mask': valid_mask
        },
        'bias_analysis': bias_analysis,
        'convergence_analysis': convergence_analysis,
        'iteration_details': iteration_results,  # NEW: include all iteration details
        'synthesis_history': synthesis_quality_history,  # NEW
        'experiment_config': {
            'move_percentage': move_percentage,
            'expansion_factor': expansion_factor,
            'synthesis_method': synthesis_method,
            'total_iterations': len(iteration_results),
            'min_sample_usage': min_sample_usage,
            'approach': 'iterative_synthetic_expansion'
        }
    }


def _analyze_convergence_synthetic(convergence_metrics):
    """Analyze convergence (same as stratified resampling function)."""
    if not convergence_metrics:
        return {'converged': False, 'reason': 'Insufficient iterations for convergence analysis'}

    # Extract convergence data
    iterations = [m['iteration'] for m in convergence_metrics]
    running_means = [m['running_r2_mean'] for m in convergence_metrics]
    running_stds = [m['running_r2_std'] for m in convergence_metrics]

    # Check for convergence (stable running mean in last 20% of iterations)
    check_start = int(len(running_means) * 0.8)
    if check_start < len(running_means) - 5:
        recent_means = running_means[check_start:]
        mean_stability = np.std(recent_means)

        converged = mean_stability < 0.01  # Convergence threshold

        return {
            'converged': converged,
            'final_mean': running_means[-1],
            'final_std': running_stds[-1],
            'stability_metric': mean_stability,
            'convergence_threshold': 0.01,
            'iterations_analyzed': len(convergence_metrics)
        }

    return {'converged': False, 'reason': 'Insufficient iterations for convergence analysis'}


def _analyze_prediction_bias(y_true: np.ndarray, y_pred: np.ndarray, y_pred_std: np.ndarray) -> Dict[str, Any]:
    """Analyze prediction bias across different target value ranges (same as stratified function)."""

    # Define target ranges
    ranges = [
        (-np.inf, -1.0, 'Very Negative'),
        (-1.0, -0.5, 'Moderate Negative'),
        (-0.5, 0.0, 'Slightly Negative'),
        (0.0, 0.5, 'Slightly Positive'),
        (0.5, 1.0, 'Moderate Positive'),
        (1.0, np.inf, 'Very Positive')
    ]

    bias_by_range = {}

    for low, high, label in ranges:
        mask = (y_true >= low) & (y_true < high)
        if np.sum(mask) > 0:
            range_true = y_true[mask]
            range_pred = y_pred[mask]
            range_std = y_pred_std[mask]

            bias = np.mean(range_pred - range_true)
            abs_bias = np.mean(np.abs(range_pred - range_true))

            from sklearn.metrics import r2_score
            range_r2 = r2_score(range_true, range_pred) if len(range_true) > 1 else np.nan

            bias_by_range[label] = {
                'n_samples': np.sum(mask),
                'bias': bias,
                'abs_bias': abs_bias,
                'r2': range_r2,
                'mean_uncertainty': np.mean(range_std),
                'range': (low, high)
            }

    return bias_by_range


def _create_iterative_synthetic_visualizations(results, save_dir):
    """Create comprehensive visualizations (enhanced version of stratified function)."""

    logger.info("Creating comprehensive visualizations...")

    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    # Create main results figure (3x3 layout for key plots)
    fig = plt.figure(figsize=(18, 15))

    # 1. Prediction Accuracy Plot
    ax1 = plt.subplot(3, 3, 1)
    y_true = results['predictions']['y_true']
    y_pred = results['predictions']['y_pred_mean']
    y_pred_std = results['predictions']['y_pred_std']

    scatter = ax1.scatter(y_true, y_pred, c=y_pred_std, cmap='viridis', alpha=0.7, s=50)

    # Perfect prediction line
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)

    ax1.set_xlabel('True Values', fontsize=18)
    ax1.set_ylabel('Mean Predicted Values', fontsize=18)
    ax1.set_title(f'Prediction Accuracy\nR² = {results["aggregated_metrics"]["r2"]:.4f}', fontsize=20)
    ax1.grid(True, alpha=0.3)

    # Add colorbar for uncertainty
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Prediction Std', fontsize=10)

    # 2. Iteration R² Distribution
    ax2 = plt.subplot(3, 3, 2)
    iteration_r2s = [result['r2'] for result in results['iteration_details']]
    ax2.hist(iteration_r2s, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.axvline(results['aggregated_metrics']['r2'], color='red', linestyle='-',
                label=f'Final R²: {results["aggregated_metrics"]["r2"]:.4f}')
    ax2.axvline(np.mean(iteration_r2s), color='blue', linestyle='--',
                label=f'Mean: {np.mean(iteration_r2s):.4f}')
    ax2.set_xlabel('R² Score', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Per-Iteration R² Distribution', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. Synthetic Sample Generation
    ax3 = plt.subplot(3, 3, 3)
    if 'iteration_details' in results:
        iterations = [r['iteration'] for r in results['iteration_details']]
        synthetic_counts = [r['synthetic_samples'] for r in results['iteration_details']]

        ax3.plot(iterations, synthetic_counts, 'g-', alpha=0.7, linewidth=1)
        ax3.axhline(y=results['synthesis_stats']['mean_synthetic_samples'], color='red', linestyle='--',
                    label=f'Mean: {results["synthesis_stats"]["mean_synthetic_samples"]:.0f}')

        ax3.set_xlabel('Iteration', fontsize=12)
        ax3.set_ylabel('Synthetic Samples Generated', fontsize=12)
        ax3.set_title('Synthetic Sample Generation', fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.legend()

    # 4. Sample Usage Statistics
    ax4 = plt.subplot(3, 3, 4)
    ax4.text(0.5, 0.7, f"Mean Usage: {results['sample_usage_stats']['mean_usage']:.1f}",
             transform=ax4.transAxes, ha='center', fontsize=12, weight='bold')
    ax4.text(0.5, 0.5,
             f"Range: {results['sample_usage_stats']['min_usage']} - {results['sample_usage_stats']['max_usage']}",
             transform=ax4.transAxes, ha='center', fontsize=11)
    ax4.text(0.5, 0.3, f"Target: {results['experiment_config']['min_sample_usage']}",
             transform=ax4.transAxes, ha='center', fontsize=11, color='red')
    ax4.set_title('Sample Usage Statistics', fontsize=14)
    ax4.axis('off')

    # 5. Expansion Ratio Analysis
    ax5 = plt.subplot(3, 3, 5)
    if 'iteration_details' in results:
        expansion_ratios = [r['expansion_ratio'] for r in results['iteration_details']]
        ax5.hist(expansion_ratios, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
        ax5.axvline(results['synthesis_stats']['mean_expansion_ratio'], color='red', linestyle='-',
                    label=f'Mean: {results["synthesis_stats"]["mean_expansion_ratio"]:.1f}x')
        ax5.set_xlabel('Expansion Ratio', fontsize=12)
        ax5.set_ylabel('Frequency', fontsize=12)
        ax5.set_title('Expansion Ratio Distribution', fontsize=14)
        ax5.grid(True, alpha=0.3)
        ax5.legend()

    # 6. Training Set Size Over Iterations
    ax6 = plt.subplot(3, 3, 6)
    if 'iteration_details' in results:
        iterations = [r['iteration'] for r in results['iteration_details']]
        training_sizes = [r['total_training_samples'] for r in results['iteration_details']]

        ax6.plot(iterations, training_sizes, 'purple', alpha=0.7, linewidth=1)
        ax6.axhline(y=np.mean(training_sizes), color='red', linestyle='--',
                    label=f'Mean: {np.mean(training_sizes):.0f}')

        ax6.set_xlabel('Iteration', fontsize=12)
        ax6.set_ylabel('Total Training Samples', fontsize=12)
        ax6.set_title('Training Set Size Over Iterations', fontsize=14)
        ax6.grid(True, alpha=0.3)
        ax6.legend()

    # 7. Residuals Analysis
    ax7 = plt.subplot(3, 3, 7)
    residuals = y_pred - y_true
    ax7.scatter(y_pred, residuals, alpha=0.7, s=30, c='orange')
    ax7.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    ax7.set_xlabel('Predicted Values', fontsize=12)
    ax7.set_ylabel('Residuals (Pred - True)', fontsize=12)
    ax7.set_title(f'Residuals Analysis\nRMSE = {results["aggregated_metrics"]["rmse"]:.4f}', fontsize=14)
    ax7.grid(True, alpha=0.3)

    # 8. Method Comparison
    ax8 = plt.subplot(3, 3, 8)
    baseline_r2 = 0.0
    stratified_r2 = 0.22
    synthetic_r2 = results['aggregated_metrics']['r2']

    methods = ['Baseline\n(Mean)', 'Stratified\nResampling', 'Iterative\nSynthetic']
    scores = [baseline_r2, stratified_r2, synthetic_r2]
    colors = ['lightgray', 'lightblue', 'lightgreen']

    bars = ax8.bar(methods, scores, color=colors, alpha=0.7)
    ax8.set_ylabel('R² Score', fontsize=12)
    ax8.set_title('Method Comparison', fontsize=14)
    ax8.set_ylim(0, max(1.0, max(scores) * 1.1))
    ax8.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                 f'{score:.4f}', ha='center', va='bottom', fontsize=11, weight='bold')

    # 9. Performance Summary
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')

    summary_text = f"""
ITERATIVE SYNTHETIC EXPANSION

Final Metrics:
• R² Score: {results['aggregated_metrics']['r2']:.4f}
• RMSE: {results['aggregated_metrics']['rmse']:.4f}
• MAE: {results['aggregated_metrics']['mae']:.4f}

Synthesis Stats:
• Method: {results['experiment_config']['synthesis_method'].title()}
• Avg Synthetic: {results['synthesis_stats']['mean_synthetic_samples']:.0f}
• Total Generated: {results['synthesis_stats']['total_synthetic_generated']:,}
• Failures: {results['synthesis_stats']['synthesis_failures']}

Convergence: {'✓ CONVERGED' if results['convergence_analysis']['converged'] else '✗ NOT CONVERGED'}
    """

    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'iterative_synthetic_validation_comprehensive.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Visualizations saved to {save_dir}")


def _save_iterative_synthetic_results(results, save_dir):
    """Save comprehensive results to files (enhanced version of stratified function)."""

    logger.info("Saving results...")

    # Save main results as JSON
    results_copy = results.copy()

    # Convert numpy arrays to lists for JSON serialization
    predictions = results_copy['predictions']
    predictions['y_true'] = predictions['y_true'].tolist()
    predictions['y_pred_mean'] = predictions['y_pred_mean'].tolist()
    predictions['y_pred_std'] = predictions['y_pred_std'].tolist()
    predictions['valid_mask'] = predictions['valid_mask'].tolist()

    with open(os.path.join(save_dir, 'iterative_synthetic_validation_results.json'), 'w') as f:
        json.dump(results_copy, f, indent=4, default=str)

    # Save detailed predictions as CSV
    y_true_array = np.array(results['predictions']['y_true'])
    y_pred_mean_array = np.array(results['predictions']['y_pred_mean'])
    y_pred_std_array = np.array(results['predictions']['y_pred_std'])

    absolute_error = np.abs(y_pred_mean_array - y_true_array)
    relative_error = np.abs(y_pred_mean_array - y_true_array) / np.abs(y_true_array)
    relative_error = np.where(np.isinf(relative_error), np.nan, relative_error)

    pred_df = pd.DataFrame({
        'sample_index': results['predictions']['sample_indices'],
        'y_true': y_true_array,
        'y_pred_mean': y_pred_mean_array,
        'y_pred_std': y_pred_std_array,
        'absolute_error': absolute_error,
        'relative_error': relative_error
    })
    pred_df.to_csv(os.path.join(save_dir, 'detailed_predictions.csv'), index=False)

    # Save iteration results (ENHANCED with synthetic info)
    iter_df = pd.DataFrame(results['iteration_details'])
    iter_df.to_csv(os.path.join(save_dir, 'iteration_results.csv'), index=False)

    # Save synthesis quality history
    if results['synthesis_history']:
        synthesis_df = pd.DataFrame(results['synthesis_history'])
        synthesis_df.to_csv(os.path.join(save_dir, 'synthesis_quality_history.csv'), index=False)

    # Save bias analysis
    if results['bias_analysis']:
        bias_df = pd.DataFrame.from_dict(results['bias_analysis'], orient='index')
        bias_df.to_csv(os.path.join(save_dir, 'bias_analysis.csv'))

    logger.info(f"All results saved to {save_dir}")


# Integration function for seamless use with retrain workflow
def integrate_iterative_synthetic_validation_in_retrain_function(
        retrain_results: Dict[str, Any],
        model_name: str,
        experiment_dir: str,
        move_percentage: float = 0.75,
        expansion_factor: int = 5,
        synthesis_method: str = 'hybrid',
        min_iterations: int = 200,
        min_sample_usage: int = 8,
        random_state: int = 42) -> Dict[str, Any]:
    """
    Integration function to add iterative synthetic expansion validation to existing retrain results.

    This function takes your existing retrain results and adds iterative synthetic expansion
    validation analysis using the validation dataset. It follows the same pattern as the
    stratified validation integration but with synthetic data expansion.

    Parameters:
    -----------
    retrain_results : Dict[str, Any]
        Results from retrain_best_model_with_gap_optimized_cv or similar retrain function
    model_name : str
        Name of the model used (for logging and file naming)
    experiment_dir : str
        Main experiment directory where results will be saved
    move_percentage : float, default=0.75
        Percentage of validation samples to move to training each iteration
        (0.75 = 75% moved, 25% used for validation each iteration)
    expansion_factor : int, default=5
        How many synthetic samples to create per moved validation sample
        (5 = 5x expansion, so 10 moved samples → 50 synthetic samples)
    synthesis_method : str, default='hybrid'
        Method for synthetic generation:
        - 'knn': KNN-based interpolation
        - 'pca': PCA eigenspace expansion
        - 'gaussian': Multivariate Gaussian modeling
        - 'hybrid': Combination of all three methods
    min_iterations : int, default=200
        Minimum number of iterations to run (will be increased if needed for sample usage)
    min_sample_usage : int, default=8
        Minimum times each validation sample should be used across iterations
    random_state : int, default=42
        Random seed for reproducibility

    Returns:
    --------
    Dict[str, Any]
        Enhanced retrain results with iterative synthetic expansion analysis

    Example Usage:
    -------------
    # After running retrain function
    retrain_results = retrain_with_explicit_control_enhanced(...)

    # Add iterative synthetic expansion validation
    enhanced_results = integrate_iterative_synthetic_validation_in_retrain_function(
        retrain_results=retrain_results,
        model_name='lightgbm',
        experiment_dir='my_experiment',
        move_percentage=0.8,      # Move 80% each iteration
        expansion_factor=7,       # 7x synthetic expansion
        synthesis_method='hybrid', # Use all synthesis methods
        min_iterations=150,       # At least 150 iterations
        min_sample_usage=10       # Each sample used 10+ times
    )
    """

    logger.info("=" * 70)
    logger.info("INTEGRATING ITERATIVE SYNTHETIC EXPANSION VALIDATION")
    logger.info("=" * 70)
    logger.info(f"Model: {model_name}")
    logger.info(f"Move percentage: {move_percentage * 100:.1f}%")
    logger.info(f"Expansion factor: {expansion_factor}x")
    logger.info(f"Synthesis method: {synthesis_method}")
    logger.info(f"Min iterations: {min_iterations}")
    logger.info(f"Min sample usage: {min_sample_usage}")

    # Extract components from retrain results
    if 'datasets' not in retrain_results:
        raise ValueError("retrain_results must contain 'datasets' key with train/validation data")

    if 'retrained_model' not in retrain_results:
        raise ValueError("retrain_results must contain 'retrained_model' key")

    datasets = retrain_results['datasets']
    retrained_model = retrain_results['retrained_model']

    # Extract required datasets
    X_train = datasets['X_train']
    y_train = datasets['y_train']
    X_validation = datasets['X_validation']
    y_validation = datasets['y_validation']

    logger.info(f"Extracted datasets:")
    logger.info(f"  Training: {len(X_train)} samples, {X_train.shape[1]} features")
    logger.info(f"  Validation: {len(X_validation)} samples")

    # Validate inputs
    if len(X_validation) < 5:
        logger.warning(f"Very small validation set ({len(X_validation)} samples). Results may be unreliable.")

    if move_percentage <= 0 or move_percentage >= 1:
        raise ValueError("move_percentage must be between 0 and 1")

    if expansion_factor < 1:
        raise ValueError("expansion_factor must be >= 1")

    if min_sample_usage < 1:
        raise ValueError("min_sample_usage must be >= 1")

    # Create iterative synthetic expansion directory
    synthetic_dir = os.path.join(experiment_dir, 'iterative_synthetic_expansion_validation')
    os.makedirs(synthetic_dir, exist_ok=True)

    # Log configuration details
    logger.info(f"Configuration validated:")
    logger.info(f"  Results directory: {synthetic_dir}")
    logger.info(
        f"  Expected synthetic samples per iteration: ~{int(len(X_validation) * move_percentage * expansion_factor)}")

    samples_per_iteration = int(len(X_validation) * move_percentage)
    estimated_total_usage = min_iterations * samples_per_iteration
    logger.info(f"  Estimated total sample usage: {estimated_total_usage}")
    logger.info(f"  Target total usage: {min_sample_usage * len(X_validation)}")

    # Run iterative synthetic expansion validation
    logger.info("Starting iterative synthetic expansion validation...")

    try:
        synthetic_results = run_iterative_synthetic_expansion_validation(
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation,
            model_template=retrained_model,
            move_percentage=move_percentage,
            expansion_factor=expansion_factor,
            synthesis_method=synthesis_method,
            min_iterations=min_iterations,
            min_sample_usage=min_sample_usage,
            random_state=random_state,
            experiment_dir=synthetic_dir
        )

        logger.info("Iterative synthetic expansion validation completed successfully!")

    except Exception as e:
        logger.error(f"Iterative synthetic expansion validation failed: {str(e)}")
        logger.error("Falling back to basic validation metrics...")

        # Fallback: just use the original validation results
        synthetic_results = {
            'aggregated_metrics': retrain_results['metrics']['validation'],
            'error': str(e),
            'fallback_used': True
        }

    # Compare with original validation results
    original_val_r2 = retrain_results['metrics']['validation']['r2']

    # Compare with stratified validation if available
    stratified_r2 = None
    if 'stratified_validation' in retrain_results:
        stratified_r2 = retrain_results['stratified_validation']['stratified_val_r2']

    # Get synthetic expansion results
    if 'error' not in synthetic_results:
        synthetic_r2 = synthetic_results['aggregated_metrics']['r2']
        synthetic_rmse = synthetic_results['aggregated_metrics']['rmse']
        synthetic_mae = synthetic_results['aggregated_metrics']['mae']

        improvement_over_original = synthetic_r2 - original_val_r2

        logger.info("=" * 50)
        logger.info("PERFORMANCE COMPARISON:")
        logger.info(f"  Original Validation R²: {original_val_r2:.4f}")

        if stratified_r2 is not None:
            improvement_over_stratified = synthetic_r2 - stratified_r2
            logger.info(f"  Stratified Validation R²: {stratified_r2:.4f}")
            logger.info(f"  Iterative Synthetic R²: {synthetic_r2:.4f}")
            logger.info(f"  Improvement over Original: {improvement_over_original:+.4f}")
            logger.info(f"  Improvement over Stratified: {improvement_over_stratified:+.4f}")
        else:
            improvement_over_stratified = None
            logger.info(f"  Iterative Synthetic R²: {synthetic_r2:.4f}")
            logger.info(f"  Improvement over Original: {improvement_over_original:+.4f}")

        logger.info(f"  Synthetic RMSE: {synthetic_rmse:.4f}")
        logger.info(f"  Synthetic MAE: {synthetic_mae:.4f}")

        # Synthesis performance summary
        if 'synthesis_stats' in synthetic_results:
            synth_stats = synthetic_results['synthesis_stats']
            logger.info("SYNTHESIS PERFORMANCE:")
            logger.info(f"  Total synthetic samples generated: {synth_stats['total_synthetic_generated']:,}")
            logger.info(f"  Average per iteration: {synth_stats['mean_synthetic_samples']:.1f}")
            logger.info(
                f"  Synthesis failures: {synth_stats['synthesis_failures']}/{synthetic_results['experiment_config']['total_iterations']}")

        # Quality assessment
        if 'synthesis_quality' in synthetic_results and synthetic_results['synthesis_quality'][
            'quality_assessments'] > 0:
            quality = synthetic_results['synthesis_quality']
            logger.info(f"  Synthesis quality score: {quality['mean_quality_score']:.3f}")

        logger.info("=" * 50)

        # Create performance comparison visualization
        _create_method_comparison_plot(
            original_val_r2, stratified_r2, synthetic_r2,
            synthetic_results, synthetic_dir, model_name
        )

        # Determine success status
        success_metrics = {
            'r2_improvement': improvement_over_original > 0.01,  # At least 1% improvement
            'convergence': synthetic_results.get('convergence_analysis', {}).get('converged', False),
            'synthesis_reliability': (synth_stats['synthesis_failures'] / synthetic_results['experiment_config'][
                'total_iterations']) < 0.2,
            'sample_usage_adequate': synthetic_results['sample_usage_stats']['min_usage'] >= min_sample_usage
        }

        overall_success = sum(success_metrics.values()) >= 3  # At least 3 out of 4 criteria met

        logger.info(f"SUCCESS CRITERIA:")
        for criterion, met in success_metrics.items():
            status = "✓" if met else "✗"
            logger.info(f"  {status} {criterion.replace('_', ' ').title()}: {met}")
        logger.info(f"Overall Success: {'✓ YES' if overall_success else '✗ NO'}")

    else:
        # Handle error case
        synthetic_r2 = original_val_r2  # Fallback value
        improvement_over_original = 0.0
        improvement_over_stratified = 0.0 if stratified_r2 is not None else None
        overall_success = False

        logger.error("Synthetic expansion validation failed. Using original validation metrics.")

    # Add iterative synthetic expansion results to retrain results
    enhancement_data = {
        'results': synthetic_results,
        'performance_comparison': {
            'original_val_r2': original_val_r2,
            'stratified_val_r2': stratified_r2,
            'synthetic_val_r2': synthetic_r2,
            'improvement_over_original': improvement_over_original,
            'improvement_over_stratified': improvement_over_stratified if stratified_r2 is not None else None
        },
        'configuration': {
            'move_percentage': move_percentage,
            'expansion_factor': expansion_factor,
            'synthesis_method': synthesis_method,
            'min_iterations': min_iterations,
            'min_sample_usage': min_sample_usage,
            'random_state': random_state
        },
        'analysis_dir': synthetic_dir,
        'success_status': overall_success,
        'integration_timestamp': datetime.now().isoformat()
    }

    # FIXED: Use consistent key name for the result storage
    retrain_results['iterative_synthetic_expansion'] = enhancement_data

    # Save enhanced results
    enhanced_results_file = os.path.join(synthetic_dir, 'enhanced_retrain_results_with_synthetic.json')
    with open(enhanced_results_file, 'w') as f:
        json.dump(retrain_results, f, indent=4, default=str)

    logger.info("=" * 70)
    logger.info("ITERATIVE SYNTHETIC EXPANSION INTEGRATION COMPLETED")
    logger.info(f"✓ Results saved to: {synthetic_dir}")
    logger.info(f"✓ Final synthetic expansion R²: {synthetic_r2:.4f}")
    logger.info(f"✓ Total improvement: {improvement_over_original:+.4f}")
    logger.info(f"✓ Overall success: {'YES' if overall_success else 'NO'}")
    logger.info("=" * 70)

    return retrain_results


def integrate_outlier_removal_into_retrain_results(
        retrain_results: Dict[str, Any],
        model_name: str,
        experiment_dir: str,
        outlier_percentage: float = 0.1,
        use_complete_revalidation: bool = True) -> Dict[str, Any]:
    """
    Simple integration function to add outlier removal to existing retrain results.

    Parameters:
    -----------
    retrain_results : Dict[str, Any]
        Your existing retrain results
    model_name : str
        Name of the model
    experiment_dir : str
        Directory to save results
    outlier_percentage : float, default=0.1
        Percentage of worst predictions to remove
    use_complete_revalidation : bool, default=True
        If True, runs complete re-validation. If False, just filters existing predictions.

    Returns:
    --------
    Dict[str, Any]
        Enhanced results with outlier analysis
    """

    logger.info(f"Integrating outlier removal for {model_name}")

    # Extract data from retrain results
    datasets = retrain_results['datasets']
    retrained_model = retrain_results['retrained_model']

    X_train = datasets['X_train']
    y_train = datasets['y_train']
    X_validation = datasets['X_validation']
    y_validation = datasets['y_validation']

    if use_complete_revalidation:
        # Run complete re-validation process
        outlier_dir = os.path.join(experiment_dir, f'{model_name}_complete_outlier_revalidation')

        complete_results = run_complete_outlier_revalidation(
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation,
            model_template=retrained_model,
            outlier_percentage=outlier_percentage,
            experiment_dir=outlier_dir
        )

        retrain_results['outlier_analysis'] = {
            'method': 'complete_revalidation',
            'results': complete_results,
            'final_r2': complete_results['comparison_metrics']['revalidation_r2'],
            'improvement': complete_results['comparison_metrics']['improvement_revalidation_vs_original'],
            'outliers_removed': complete_results['comparison_metrics']['outliers_removed']
        }

    else:
        # Simple outlier filtering on existing predictions
        y_pred = retrained_model.predict(X_validation)

        analysis_dir = os.path.join(experiment_dir, f'{model_name}_outlier_filtering')

        y_true_filtered, y_pred_filtered, X_val_filtered, remaining_indices, outliers_df = identify_and_remove_worst_outliers(
            y_true=np.array(y_validation.values),
            y_pred=y_pred,
            X_validation=X_validation,
            sample_indices=X_validation.index.tolist(),
            outlier_percentage=outlier_percentage,
            save_dir=analysis_dir
        )

        from sklearn.metrics import r2_score
        original_r2 = r2_score(y_validation, y_pred)
        filtered_r2 = r2_score(y_true_filtered, y_pred_filtered)

        retrain_results['outlier_analysis'] = {
            'method': 'simple_filtering',
            'original_r2': original_r2,
            'filtered_r2': filtered_r2,
            'improvement': filtered_r2 - original_r2,
            'outliers_removed': len(outliers_df),
            'outliers_data': outliers_df.to_dict('records')
        }

    logger.info(f"Outlier analysis completed for {model_name}")
    return retrain_results

def _create_method_comparison_plot(original_r2, stratified_r2, synthetic_r2,
                                   synthetic_results, save_dir, model_name):
    """Create visualization comparing different validation methods."""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: R² Comparison
    methods = ['Original\nValidation']
    scores = [original_r2]
    colors = ['lightgray']

    if stratified_r2 is not None:
        methods.append('Stratified\nResampling')
        scores.append(stratified_r2)
        colors.append('lightblue')

    methods.append('Iterative\nSynthetic')
    scores.append(synthetic_r2)
    colors.append('lightgreen')

    bars = ax1.bar(methods, scores, color=colors, alpha=0.8)
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title(f'{model_name.upper()} - Validation Method Comparison', fontsize=14, weight='bold')
    ax1.set_ylim(0, max(1.0, max(scores) * 1.1))
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                 f'{score:.4f}', ha='center', va='bottom', fontsize=11, weight='bold')

    # Add improvement annotations
    if len(scores) >= 2:
        improvement = scores[-1] - scores[0]
        ax1.text(0.5, 0.95, f'Total Improvement: {improvement:+.4f}',
                 transform=ax1.transAxes, ha='center', va='top',
                 bbox=dict(boxstyle='round', facecolor='yellow' if improvement > 0 else 'lightcoral', alpha=0.7),
                 fontsize=12, weight='bold')

    # Plot 2: Synthesis Statistics (if available)
    ax2.axis('off')
    if 'synthesis_stats' in synthetic_results:
        synth_stats = synthetic_results['synthesis_stats']
        config = synthetic_results['experiment_config']

        stats_text = f"""
SYNTHESIS STATISTICS

Configuration:
• Method: {config['synthesis_method'].title()}
• Expansion Factor: {config['expansion_factor']}x
• Iterations: {config['total_iterations']}

Generation Results:
• Total Synthetic: {synth_stats['total_synthetic_generated']:,}
• Avg per Iteration: {synth_stats['mean_synthetic_samples']:.0f}
• Success Rate: {(1 - synth_stats['synthesis_failures'] / config['total_iterations']) * 100:.1f}%

Quality:
• Mean Expansion: {synth_stats['mean_expansion_ratio']:.1f}x
• Range: {synth_stats['min_synthetic_samples']}-{synth_stats['max_synthetic_samples']}
        """

        ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax2.set_title('Synthesis Performance', fontsize=14, weight='bold')

    # Plot 3: Sample Usage Analysis (if available)
    if 'sample_usage_stats' in synthetic_results:
        usage_stats = synthetic_results['sample_usage_stats']

        # Create usage summary
        usage_data = ['Min Usage', 'Mean Usage', 'Max Usage', 'Target Usage']
        usage_values = [
            usage_stats['min_usage'],
            usage_stats['mean_usage'],
            usage_stats['max_usage'],
            synthetic_results['experiment_config']['min_sample_usage']
        ]
        usage_colors = ['lightcoral', 'lightblue', 'lightgreen', 'red']

        bars = ax3.bar(usage_data, usage_values, color=usage_colors, alpha=0.7)
        ax3.set_ylabel('Usage Count', fontsize=12)
        ax3.set_title('Sample Usage Analysis', fontsize=14, weight='bold')
        ax3.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, value in zip(bars, usage_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2, height + 0.1,
                     f'{value:.1f}', ha='center', va='bottom', fontsize=10, weight='bold')

        # Add target line
        target_usage = synthetic_results['experiment_config']['min_sample_usage']
        ax3.axhline(y=target_usage, color='red', linestyle='--', alpha=0.8, linewidth=2)

    # Plot 4: Performance Summary
    ax4.axis('off')

    # Calculate summary metrics
    final_r2 = synthetic_results['aggregated_metrics']['r2']
    final_rmse = synthetic_results['aggregated_metrics']['rmse']
    final_mae = synthetic_results['aggregated_metrics']['mae']

    improvement_over_original = final_r2 - original_r2

    convergence_status = synthetic_results.get('convergence_analysis', {}).get('converged', False)

    summary_text = f"""
PERFORMANCE SUMMARY

Final Metrics:
• R² Score: {final_r2:.4f}
• RMSE: {final_rmse:.4f}
• MAE: {final_mae:.4f}

Improvements:
• vs Original: {improvement_over_original:+.4f}
"""

    if stratified_r2 is not None:
        improvement_over_stratified = final_r2 - stratified_r2
        summary_text += f"• vs Stratified: {improvement_over_stratified:+.4f}\n"

    summary_text += f"""
Process Status:
• Convergence: {'✓ YES' if convergence_status else '✗ NO'}
• Valid Predictions: {synthetic_results['aggregated_metrics']['valid_predictions']}/{synthetic_results['aggregated_metrics']['total_samples']}

Overall: {'SUCCESS' if improvement_over_original > 0 else 'MIXED RESULTS'}
    """

    # Color based on performance
    bg_color = 'lightgreen' if improvement_over_original > 0.01 else 'lightyellow' if improvement_over_original > 0 else 'lightcoral'

    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.8))
    ax4.set_title('Overall Assessment', fontsize=14, weight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_iterative_synthetic_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Method comparison plot saved for {model_name}")


# Convenience function for quick experimentation
def quick_synthetic_expansion_experiment(retrain_results, model_name, experiment_dir,
                                         expansion_factors=[3, 5, 7],
                                         synthesis_methods=['knn', 'hybrid'],
                                         move_percentages=[0.7, 0.8]):
    """
    Run multiple synthetic expansion experiments with different parameters.

    This function allows you to quickly test different combinations of parameters
    to find the optimal configuration for your specific dataset.

    Parameters:
    -----------
    retrain_results : dict
        Results from retrain function
    model_name : str
        Name of the model
    experiment_dir : str
        Base experiment directory
    expansion_factors : list, default=[3, 5, 7]
        List of expansion factors to test
    synthesis_methods : list, default=['knn', 'hybrid']
        List of synthesis methods to test
    move_percentages : list, default=[0.7, 0.8]
        List of move percentages to test

    Returns:
    --------
    dict
        Results for all parameter combinations with best configuration identified
    """

    logger.info("=" * 70)
    logger.info("QUICK SYNTHETIC EXPANSION EXPERIMENT")
    logger.info("=" * 70)
    logger.info(f"Testing {len(expansion_factors)} expansion factors")
    logger.info(f"Testing {len(synthesis_methods)} synthesis methods")
    logger.info(f"Testing {len(move_percentages)} move percentages")

    total_experiments = len(expansion_factors) * len(synthesis_methods) * len(move_percentages)
    logger.info(f"Total experiments: {total_experiments}")

    experiment_results = {}
    best_config = None
    best_r2 = -np.inf

    experiment_count = 0

    for expansion_factor in expansion_factors:
        for synthesis_method in synthesis_methods:
            for move_percentage in move_percentages:
                experiment_count += 1

                config_name = f"exp{expansion_factor}_{synthesis_method}_move{int(move_percentage * 100)}"
                logger.info(f"Running experiment {experiment_count}/{total_experiments}: {config_name}")

                try:
                    # Run experiment with current configuration
                    exp_dir = os.path.join(experiment_dir, f'synthetic_experiments/{config_name}')

                    enhanced_results = integrate_iterative_synthetic_validation_in_retrain_function(
                        retrain_results=retrain_results.copy(),  # Use copy to avoid modifying original
                        model_name=model_name,
                        experiment_dir=exp_dir,
                        move_percentage=move_percentage,
                        expansion_factor=expansion_factor,
                        synthesis_method=synthesis_method,
                        min_iterations=50,  # Reduced for quick experiments
                        min_sample_usage=3,  # Reduced for quick experiments
                        random_state=42
                    )

                    # Extract key results
                    synthetic_data = enhanced_results['iterative_synthetic_expansion']
                    final_r2 = synthetic_data['performance_comparison']['synthetic_val_r2']
                    improvement = synthetic_data['performance_comparison']['improvement_over_original']

                    experiment_results[config_name] = {
                        'config': {
                            'expansion_factor': expansion_factor,
                            'synthesis_method': synthesis_method,
                            'move_percentage': move_percentage
                        },
                        'r2': final_r2,
                        'improvement': improvement,
                        'success': synthetic_data['success_status'],
                        'full_results': enhanced_results
                    }

                    # Track best configuration
                    if final_r2 > best_r2:
                        best_r2 = final_r2
                        best_config = config_name

                    logger.info(f"  R²: {final_r2:.4f}, Improvement: {improvement:+.4f}")

                except Exception as e:
                    logger.error(f"  Experiment {config_name} failed: {str(e)}")
                    experiment_results[config_name] = {
                        'config': {
                            'expansion_factor': expansion_factor,
                            'synthesis_method': synthesis_method,
                            'move_percentage': move_percentage
                        },
                        'r2': -np.inf,
                        'improvement': -np.inf,
                        'success': False,
                        'error': str(e)
                    }

    # Summary analysis
    logger.info("=" * 50)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 50)

    if best_config:
        logger.info(f"Best configuration: {best_config}")
        logger.info(f"Best R²: {best_r2:.4f}")
        logger.info(f"Best config details: {experiment_results[best_config]['config']}")

    # Sort results by R²
    sorted_results = sorted(experiment_results.items(), key=lambda x: x[1]['r2'], reverse=True)

    logger.info("Top 3 configurations:")
    for i, (config_name, results) in enumerate(sorted_results[:3]):
        logger.info(f"  {i + 1}. {config_name}: R² = {results['r2']:.4f}")

    return {
        'experiment_results': experiment_results,
        'best_config': best_config,
        'best_r2': best_r2,
        'sorted_results': sorted_results,
        'summary': {
            'total_experiments': total_experiments,
            'successful_experiments': sum(1 for r in experiment_results.values() if r['success']),
            'best_improvement': experiment_results[best_config]['improvement'] if best_config else None
        }
    }


# Easy access function for results
def get_synthetic_expansion_r2(retrain_results: Dict[str, Any]) -> float:
    """
    Easy function to extract the final R² score from synthetic expansion results.

    This is a helper function to avoid KeyError issues when accessing nested results.

    Parameters:
    -----------
    retrain_results : Dict[str, Any]
        Results from retrain function with synthetic expansion analysis

    Returns:
    --------
    float
        Final R² score from synthetic expansion, or original validation R² if not available
    """
    try:
        # Try to get synthetic expansion R²
        return retrain_results['iterative_synthetic_expansion']['performance_comparison']['synthetic_val_r2']
    except KeyError:
        try:
            # Fallback to original validation R²
            return retrain_results['metrics']['validation']['r2']
        except KeyError:
            # Last resort fallback
            logger.warning("Could not find R² score in results, returning 0.0")
            return 0.0


def test_synthetic_expansion():
    """Test function to verify synthetic expansion works with dummy data."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.datasets import make_regression

    # Create dummy data that mimics your soil science problem
    X, y = make_regression(n_samples=200, n_features=10, noise=0.1, random_state=42)

    # Add some domain shift to validation data
    X_val_shifted = X[-30:] + np.random.normal(0, 0.5, (30, 10))  # Shift validation domain
    y_val_shifted = y[-30:] + np.random.normal(0, 0.3, 30)  # Add target noise

    X_df = pd.DataFrame(X[:-30], columns=[f'feature_{i}' for i in range(10)])
    y_series = pd.Series(y[:-30], name='target')
    X_val_df = pd.DataFrame(X_val_shifted, columns=[f'feature_{i}' for i in range(10)])
    y_val_series = pd.Series(y_val_shifted, name='target')

    # Test synthetic expansion
    model_template = RandomForestRegressor(n_estimators=50, random_state=42)

    results = run_iterative_synthetic_expansion_validation(
        X_train=X_df,
        y_train=y_series,
        X_validation=X_val_df,
        y_validation=y_val_series,
        model_template=model_template,
        move_percentage=0.8,
        expansion_factor=3,  # Smaller for testing
        synthesis_method='hybrid',
        min_iterations=20,  # Smaller for testing
        min_sample_usage=2,
        experiment_dir='test_synthetic_expansion'
    )

    print(f"Test completed! Final R²: {results['aggregated_metrics']['r2']:.4f}")
    print(f"Synthetic quality: {results['synthesis_stats']['mean_synthetic_samples']:.1f} samples/iteration")

    return results

