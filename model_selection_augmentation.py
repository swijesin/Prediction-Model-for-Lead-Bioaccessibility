import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from typing import Tuple, Dict, Optional
import logging
import os
import json

logger = logging.getLogger(__name__)


class KNNDataAugmentor:
    """
    Advanced data augmentation for model selection phase.

    Supports multiple synthesis methods:
    - 'knn': KNN-based interpolation (default)
    - 'pca': PCA eigenspace expansion
    - 'gaussian': Multivariate Gaussian modeling
    - 'hybrid': Combination of all three methods
    """

    def __init__(self,
                 n_neighbors: int = 5,
                 expansion_factor: float = 1.5,
                 noise_level: float = 0.05,
                 synthesis_method: str = 'knn',
                 random_state: int = 42):
        """
        Initialize data augmentor with multiple synthesis methods.

        Parameters:
        -----------
        n_neighbors : int, default=5
            Number of neighbors to consider for KNN interpolation
        expansion_factor : float, default=1.5
            Multiplier for dataset size (1.5 = 50% more samples)
            Use 1.0 for no augmentation
        noise_level : float, default=0.05
            Amount of noise to add (as fraction of local variance)
            Lower = more conservative, Higher = more diverse
        synthesis_method : str, default='knn'
            Method for generating synthetic samples:
            - 'knn': KNN-based interpolation
            - 'pca': PCA eigenspace expansion
            - 'gaussian': Multivariate Gaussian modeling
            - 'hybrid': Combination of all three methods
        random_state : int, default=42
            Random seed for reproducibility
        """
        self.n_neighbors = n_neighbors
        self.expansion_factor = expansion_factor
        self.noise_level = noise_level
        self.synthesis_method = synthesis_method.lower()
        self.random_state = random_state

        # Validate synthesis method
        valid_methods = ['knn', 'pca', 'gaussian', 'hybrid']
        if self.synthesis_method not in valid_methods:
            raise ValueError(f"synthesis_method must be one of {valid_methods}, got '{synthesis_method}'")

        self.is_fitted = False
        self.feature_names = None
        self.target_name = None

        # Components for different synthesis methods
        self.scaler = StandardScaler()
        self.pca = PCA()
        self.knn = None
        self.eigenvalues_ = None
        self.eigenvectors_ = None

        np.random.seed(random_state)

        logger.info(f"Initialized Data Augmentor:")
        logger.info(f"  Synthesis method: {self.synthesis_method}")
        logger.info(f"  Expansion factor: {expansion_factor}x")
        logger.info(f"  Neighbors (KNN): {n_neighbors}")
        logger.info(f"  Noise level: {noise_level}")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'KNNDataAugmentor':
        """
        Fit the augmentor (stores feature names and fits PCA if needed).

        Parameters:
        -----------
        X : pd.DataFrame
            Training features
        y : pd.Series
            Training targets

        Returns:
        --------
        self
        """
        self.feature_names = X.columns.tolist()
        self.target_name = y.name if hasattr(y, 'name') else 'target'

        # Fit components based on synthesis method
        if self.synthesis_method in ['pca', 'gaussian', 'hybrid']:
            # Scale data for PCA/Gaussian
            X_scaled = self.scaler.fit_transform(X)

            if self.synthesis_method in ['pca', 'hybrid']:
                # Fit PCA
                self.pca.fit(X_scaled)
                self.eigenvalues_ = self.pca.explained_variance_
                self.eigenvectors_ = self.pca.components_
                logger.info(f"  PCA fitted: {len(self.eigenvalues_)} components, "
                            f"{np.sum(self.pca.explained_variance_ratio_[:5]):.2%} variance in top 5")

        self.is_fitted = True
        logger.info(f"Augmentor fitted on {len(X)} samples with {len(self.feature_names)} features")

        return self

    def augment(self,
                X: pd.DataFrame,
                y: pd.Series,
                return_augmented_only: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate synthetic samples using the specified method.

        Parameters:
        -----------
        X : pd.DataFrame
            Original training features
        y : pd.Series
            Original training targets
        return_augmented_only : bool, default=False
            If True, return only synthetic samples
            If False, return original + synthetic samples

        Returns:
        --------
        X_augmented : pd.DataFrame
            Augmented feature matrix
        y_augmented : pd.Series
            Augmented target values
        """
        if not self.is_fitted:
            raise ValueError("Augmentor must be fitted before augmentation")

        # Calculate number of synthetic samples to generate
        n_original = len(X)
        n_synthetic = int(n_original * (self.expansion_factor - 1))

        if n_synthetic <= 0:
            logger.info("Expansion factor <= 1.0, no augmentation performed")
            return X.copy(), y.copy()

        logger.info(f"Generating {n_synthetic} synthetic samples using {self.synthesis_method} method...")

        # Scale data
        X_scaled = self.scaler.transform(X)

        # Generate synthetic samples based on method
        if self.synthesis_method == 'knn':
            X_synthetic_scaled, y_synthetic = self._generate_knn_samples(X_scaled, y, n_synthetic)
        elif self.synthesis_method == 'pca':
            X_synthetic_scaled, y_synthetic = self._generate_pca_samples(X_scaled, y, n_synthetic)
        elif self.synthesis_method == 'gaussian':
            X_synthetic_scaled, y_synthetic = self._generate_gaussian_samples(X_scaled, y, n_synthetic)
        elif self.synthesis_method == 'hybrid':
            X_synthetic_scaled, y_synthetic = self._generate_hybrid_samples(X_scaled, y, n_synthetic)

        # Transform back to original scale
        X_synthetic = pd.DataFrame(
            self.scaler.inverse_transform(X_synthetic_scaled),
            columns=self.feature_names
        )

        logger.info(f"Successfully generated {len(X_synthetic)} synthetic samples")

        # Return based on flag
        if return_augmented_only:
            return X_synthetic, y_synthetic
        else:
            # Combine original and synthetic
            X_augmented = pd.concat([X, X_synthetic], ignore_index=True)
            y_augmented = pd.concat([y, y_synthetic], ignore_index=True)

            logger.info(f"Final augmented dataset: {len(X_augmented)} samples "
                        f"({n_original} original + {n_synthetic} synthetic)")

            return X_augmented, y_augmented

    def _generate_knn_samples(self, X_scaled: np.ndarray, y: pd.Series,
                              n_synthetic: int) -> Tuple[np.ndarray, pd.Series]:
        """Generate samples using KNN-based interpolation."""
        logger.info("Using KNN-based synthesis...")

        n_original = len(X_scaled)

        # Fit KNN on original data
        n_neighbors_actual = min(self.n_neighbors, n_original - 1)
        self.knn = NearestNeighbors(n_neighbors=n_neighbors_actual + 1)  # +1 includes self
        self.knn.fit(X_scaled)

        # Generate synthetic samples
        X_synthetic_list = []
        y_synthetic_list = []

        for i in range(n_synthetic):
            # Randomly select an anchor point
            anchor_idx = np.random.randint(0, n_original)
            anchor_x = X_scaled[anchor_idx]
            anchor_y = y.iloc[anchor_idx]

            # Find neighbors of the anchor
            distances, neighbor_indices = self.knn.kneighbors([anchor_x])

            # Exclude the anchor itself (first neighbor)
            neighbor_indices = neighbor_indices[0][1:]

            if len(neighbor_indices) > 0:
                # Select a random neighbor for interpolation
                neighbor_idx_local = np.random.randint(0, len(neighbor_indices))
                neighbor_idx = neighbor_indices[neighbor_idx_local]
                neighbor_x = X_scaled[neighbor_idx]
                neighbor_y = y.iloc[neighbor_idx]

                # Interpolation weight (Beta distribution biases toward anchor)
                alpha = np.random.beta(2, 1)  # Favors values closer to 1 (anchor)

                # Create synthetic sample in feature space
                synthetic_x = alpha * anchor_x + (1 - alpha) * neighbor_x

                # Add controlled noise based on local neighborhood variance
                local_samples = X_scaled[neighbor_indices]
                local_std = np.std(local_samples, axis=0)
                noise = np.random.normal(0, local_std * self.noise_level)
                synthetic_x = synthetic_x + noise

                # Create synthetic target
                synthetic_y = alpha * anchor_y + (1 - alpha) * neighbor_y

                # Add smaller noise to target
                target_std = np.std(y)
                target_noise = np.random.normal(0, target_std * self.noise_level * 0.5)
                synthetic_y = synthetic_y + target_noise

                X_synthetic_list.append(synthetic_x)
                y_synthetic_list.append(synthetic_y)
            else:
                # Fallback: just add noise to anchor
                synthetic_x = anchor_x + np.random.normal(0, np.std(X_scaled, axis=0) * self.noise_level)
                synthetic_y = anchor_y + np.random.normal(0, np.std(y) * self.noise_level * 0.5)

                X_synthetic_list.append(synthetic_x)
                y_synthetic_list.append(synthetic_y)

        X_synthetic = np.array(X_synthetic_list)
        y_synthetic = pd.Series(y_synthetic_list, name=self.target_name)

        return X_synthetic, y_synthetic

    def _generate_pca_samples(self, X_scaled: np.ndarray, y: pd.Series,
                              n_synthetic: int) -> Tuple[np.ndarray, pd.Series]:
        """Generate samples using PCA-based expansion in eigenspace."""
        logger.info("Using PCA-based synthesis...")

        # Safety checks
        if X_scaled.shape[0] < 2:
            logger.warning("Too few samples for PCA synthesis, using KNN instead")
            return self._generate_knn_samples(X_scaled, y, n_synthetic)

        if X_scaled.shape[1] == 0:
            logger.warning("No features available for PCA synthesis, using KNN instead")
            return self._generate_knn_samples(X_scaled, y, n_synthetic)

        # Check if PCA was properly fitted
        if not hasattr(self.pca, 'components_') or self.pca.components_.shape[0] == 0:
            logger.warning("PCA not properly fitted, using KNN instead")
            return self._generate_knn_samples(X_scaled, y, n_synthetic)

        try:
            # Transform to PCA space
            X_pca = self.pca.transform(X_scaled)

            # Generate samples in PCA space
            X_synthetic_pca = []
            y_synthetic_list = []

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
                    y_synthetic_list.append(synthetic_target)

            # Transform back to original feature space
            X_synthetic_pca = np.array(X_synthetic_pca)
            X_synthetic = self.pca.inverse_transform(X_synthetic_pca)
            y_synthetic = pd.Series(y_synthetic_list, name=self.target_name) if y is not None else None

            return X_synthetic, y_synthetic

        except Exception as e:
            logger.warning(f"PCA synthesis failed: {str(e)}, falling back to KNN")
            return self._generate_knn_samples(X_scaled, y, n_synthetic)

    def _generate_gaussian_samples(self, X_scaled: np.ndarray, y: pd.Series,
                                   n_synthetic: int) -> Tuple[np.ndarray, pd.Series]:
        """Generate samples using multivariate Gaussian modeling."""
        logger.info("Using Gaussian-based synthesis...")

        # Safety checks
        if len(X_scaled) < 2:
            logger.warning("Too few samples for Gaussian synthesis, using KNN instead")
            return self._generate_knn_samples(X_scaled, y, n_synthetic)

        if X_scaled.shape[1] == 0:
            logger.warning("No features available for Gaussian synthesis, using KNN instead")
            return self._generate_knn_samples(X_scaled, y, n_synthetic)

        try:
            # Fit multivariate Gaussian to the data
            mean = np.mean(X_scaled, axis=0)
            cov = np.cov(X_scaled.T)

            # Check if covariance matrix is valid
            if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
                logger.warning("Invalid covariance matrix, using KNN instead")
                return self._generate_knn_samples(X_scaled, y, n_synthetic)

            # Generate synthetic samples
            if n_synthetic == 0:
                n_synthetic = 1

            X_synthetic = np.random.multivariate_normal(mean, cov, n_synthetic)

            # Generate synthetic targets
            y_synthetic = None
            if y is not None and len(y) > 0:
                # Fit simple linear relationship between features and targets for synthesis
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
                    y_synthetic = pd.Series(y_pred + y_noise, name=self.target_name)
                else:
                    # If too few samples for regression, just add noise to existing targets
                    y_synthetic_list = []
                    target_std = np.std(y) if len(y) > 1 else 0.1
                    for _ in range(n_synthetic):
                        ref_idx = np.random.randint(0, len(y))
                        noise = np.random.normal(0, target_std * self.noise_level)
                        y_synthetic_list.append(y.iloc[ref_idx] + noise)
                    y_synthetic = pd.Series(y_synthetic_list, name=self.target_name)

            return X_synthetic, y_synthetic

        except Exception as e:
            logger.warning(f"Gaussian synthesis failed: {str(e)}, falling back to KNN")
            return self._generate_knn_samples(X_scaled, y, n_synthetic)

    def _generate_hybrid_samples(self, X_scaled: np.ndarray, y: pd.Series,
                                 n_synthetic: int) -> Tuple[np.ndarray, pd.Series]:
        """Generate samples using hybrid approach combining multiple methods."""
        logger.info("Using hybrid synthesis (KNN + PCA + Gaussian)...")

        if len(X_scaled) < 3:
            logger.warning("Too few samples for hybrid synthesis, using KNN only")
            return self._generate_knn_samples(X_scaled, y, n_synthetic)

        # Divide synthetic samples among three methods
        n_per_method = n_synthetic // 3
        remainder = n_synthetic % 3

        # Method 1: KNN (gets any remainder samples)
        n_knn = n_per_method + remainder
        X_knn, y_knn = self._generate_knn_samples(X_scaled, y, n_knn)

        # Method 2: PCA
        X_pca, y_pca = self._generate_pca_samples(X_scaled, y, n_per_method)

        # Method 3: Gaussian
        X_gauss, y_gauss = self._generate_gaussian_samples(X_scaled, y, n_per_method)

        # Combine all synthetic samples
        X_synthetic = np.vstack([X_knn, X_pca, X_gauss])

        if y is not None:
            y_synthetic = pd.concat([
                y_knn.reset_index(drop=True),
                y_pca.reset_index(drop=True),
                y_gauss.reset_index(drop=True)
            ], ignore_index=True)
            y_synthetic.name = self.target_name
        else:
            y_synthetic = None

        logger.info(f"Hybrid synthesis: {len(X_knn)} KNN + {len(X_pca)} PCA + {len(X_gauss)} Gaussian")

        return X_synthetic, y_synthetic

    def fit_augment(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit and augment in one step.

        Parameters:
        -----------
        X : pd.DataFrame
            Training features
        y : pd.Series
            Training targets

        Returns:
        --------
        X_augmented : pd.DataFrame
            Augmented features
        y_augmented : pd.Series
            Augmented targets
        """
        return self.fit(X, y).augment(X, y)

    def get_augmentation_stats(self, X_original: pd.DataFrame,
                               X_augmented: pd.DataFrame,
                               y_original: pd.Series,
                               y_augmented: pd.Series) -> Dict:
        """
        Calculate statistics comparing original and augmented datasets.

        Parameters:
        -----------
        X_original : pd.DataFrame
            Original features
        X_augmented : pd.DataFrame
            Augmented features
        y_original : pd.Series
            Original targets
        y_augmented : pd.Series
            Augmented targets

        Returns:
        --------
        Dict
            Statistics comparing datasets
        """
        stats = {
            'synthesis_method': self.synthesis_method,
            'original_samples': len(X_original),
            'augmented_samples': len(X_augmented),
            'synthetic_samples': len(X_augmented) - len(X_original),
            'expansion_achieved': len(X_augmented) / len(X_original),
            'n_neighbors': self.n_neighbors,
            'noise_level': self.noise_level,
            'feature_stats': {},
            'target_stats': {}
        }

        # Compare feature distributions
        for col in X_original.columns:
            stats['feature_stats'][col] = {
                'original_mean': X_original[col].mean(),
                'augmented_mean': X_augmented[col].mean(),
                'original_std': X_original[col].std(),
                'augmented_std': X_augmented[col].std(),
                'mean_shift': abs(X_augmented[col].mean() - X_original[col].mean()),
                'mean_shift_pct': abs(X_augmented[col].mean() - X_original[col].mean()) / (
                            abs(X_original[col].mean()) + 1e-8) * 100
            }

        # Compare target distributions
        stats['target_stats'] = {
            'original_mean': y_original.mean(),
            'augmented_mean': y_augmented.mean(),
            'original_std': y_original.std(),
            'augmented_std': y_augmented.std(),
            'original_range': [y_original.min(), y_original.max()],
            'augmented_range': [y_augmented.min(), y_augmented.max()],
            'mean_shift': abs(y_augmented.mean() - y_original.mean()),
            'mean_shift_pct': abs(y_augmented.mean() - y_original.mean()) / (abs(y_original.mean()) + 1e-8) * 100
        }

        # Add method-specific stats
        if self.synthesis_method in ['pca', 'hybrid']:
            if hasattr(self, 'eigenvalues_') and self.eigenvalues_ is not None:
                stats['pca_stats'] = {
                    'n_components': len(self.eigenvalues_),
                    'explained_variance_ratio': self.pca.explained_variance_ratio_.tolist() if hasattr(self.pca,
                                                                                                       'explained_variance_ratio_') else None,
                    'cumulative_variance_5': np.sum(self.pca.explained_variance_ratio_[:5]) if hasattr(self.pca,
                                                                                                       'explained_variance_ratio_') else None
                }

        return stats

    def save_augmentation_report(self, stats: Dict, save_dir: str, prefix: str = '') -> None:
        """
        Save augmentation statistics to file.

        Parameters:
        -----------
        stats : Dict
            Augmentation statistics from get_augmentation_stats()
        save_dir : str
            Directory to save report
        prefix : str, optional
            Prefix for filename
        """
        os.makedirs(save_dir, exist_ok=True)

        filename = f'{prefix}augmentation_stats_{self.synthesis_method}.json' if prefix else f'augmentation_stats_{self.synthesis_method}.json'
        filepath = os.path.join(save_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=4, default=str)

        logger.info(f"Augmentation report saved to {filepath}")