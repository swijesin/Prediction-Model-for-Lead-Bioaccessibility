import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.base import clone
import joblib
import os
import json
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from validationanalyzer import analyze_validation_performance



# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StratifiedValidationResampler:
    """
    Advanced validation resampler using bootstrap aggregation with domain adaptation.

    This class implements the sophisticated approach you described:
    1. Move 70% of validation samples to training
    2. Train model and predict on remaining 30%
    3. Repeat 100+ times with different random selections
    4. Ensure each validation sample is used 5+ times
    5. Average predictions and analyze results
    """

    def __init__(self,
                 move_percentage: float = 0.7,
                 min_iterations: int = 100,
                 min_sample_usage: int = 5,
                 random_state: int = 42,
                 experiment_dir: str = None):
        """
        Initialize the resampler.

        Parameters:
        -----------
        move_percentage : float, default=0.7
            Percentage of validation samples to move to training each iteration
        min_iterations : int, default=100
            Minimum number of iterations to run
        min_sample_usage : int, default=5
            Minimum times each validation sample should be used
        random_state : int, default=42
            Random seed for reproducibility
        experiment_dir : str, optional
            Directory to save results and visualizations
        """
        self.move_percentage = move_percentage
        self.min_iterations = min_iterations
        self.min_sample_usage = min_sample_usage
        self.random_state = random_state
        self.experiment_dir = experiment_dir or 'stratified_validation_results'

        # Create experiment directory
        os.makedirs(self.experiment_dir, exist_ok=True)

        # Results storage
        self.iteration_results = []
        self.sample_predictions = {}
        self.sample_usage_count = {}
        self.convergence_metrics = []

        # Model and preprocessing components
        self.base_model = None
        self.preprocessor = None
        self.selected_features = None

        logger.info(f"Initialized StratifiedValidationResampler:")
        logger.info(f"  Move percentage: {move_percentage * 100:.1f}%")
        logger.info(f"  Min iterations: {min_iterations}")
        logger.info(f"  Min sample usage: {min_sample_usage}")
        logger.info(f"  Results dir: {self.experiment_dir}")

    def calculate_required_iterations(self, n_validation_samples: int) -> int:
        """Calculate the number of iterations needed to meet sample usage requirements."""
        samples_per_iteration = int(n_validation_samples * self.move_percentage)
        required_iterations = int(np.ceil((self.min_sample_usage * n_validation_samples) / samples_per_iteration))

        # Ensure we meet minimum iterations
        required_iterations = max(required_iterations, self.min_iterations)

        logger.info(f"Calculated {required_iterations} iterations needed for {n_validation_samples} validation samples")
        logger.info(f"  Samples per iteration: {samples_per_iteration}")
        logger.info(f"  Target total usage: {self.min_sample_usage * n_validation_samples}")

        return required_iterations

    def run_validation_resampling(self,
                                  X_train: pd.DataFrame,
                                  y_train: pd.Series,
                                  X_validation: pd.DataFrame,
                                  y_validation: pd.Series,
                                  model_template: Any,
                                  preprocessor: Any = None,
                                  selected_features: List[str] = None) -> Dict[str, Any]:
        """
        Run the main validation resampling procedure.

        Parameters:
        -----------
        X_train : pd.DataFrame
            Original training features
        y_train : pd.Series
            Original training targets
        X_validation : pd.DataFrame
            Validation features
        y_validation : pd.Series
            Validation targets
        model_template : sklearn estimator
            Model to use for training (will be cloned each iteration)
        preprocessor : optional
            Preprocessing pipeline to apply
        selected_features : List[str], optional
            List of features to use (if None, uses all)

        Returns:
        --------
        Dict[str, Any]
            Comprehensive results dictionary
        """
        logger.info("=" * 60)
        logger.info("STARTING STRATIFIED VALIDATION RESAMPLING")
        logger.info("=" * 60)

        # Store components
        self.base_model = model_template
        self.preprocessor = preprocessor
        self.selected_features = selected_features or list(X_train.columns)

        # Prepare data
        n_validation = len(X_validation)
        validation_indices = X_validation.index.tolist()

        # Calculate required iterations
        required_iterations = self.calculate_required_iterations(n_validation)

        # Initialize tracking
        self.sample_predictions = {idx: [] for idx in validation_indices}
        self.sample_usage_count = {idx: 0 for idx in validation_indices}
        self.iteration_results = []
        self.convergence_metrics = []

        logger.info(f"Starting {required_iterations} iterations...")
        logger.info(f"Validation samples: {n_validation}")
        logger.info(f"Training samples: {len(X_train)}")

        # Run iterations
        np.random.seed(self.random_state)

        for iteration in range(required_iterations):
            if iteration % 20 == 0:
                logger.info(f"Iteration {iteration + 1}/{required_iterations}")

            # Random selection of validation samples to move to training
            n_move = int(n_validation * self.move_percentage)
            move_indices = np.random.choice(validation_indices, size=n_move, replace=False)
            remain_indices = [idx for idx in validation_indices if idx not in move_indices]

            # Update usage counts
            for idx in move_indices:
                self.sample_usage_count[idx] += 1

            # Create augmented training set
            X_train_aug = pd.concat([X_train, X_validation.loc[move_indices]])
            y_train_aug = pd.concat([y_train, y_validation.loc[move_indices]])

            # Validation subset for this iteration
            X_val_iter = X_validation.loc[remain_indices]
            y_val_iter = y_validation.loc[remain_indices]

            # Apply preprocessing if provided
            if self.preprocessor is not None:
                X_train_processed = self.preprocessor.transform(X_train_aug)
                X_val_processed = self.preprocessor.transform(X_val_iter)

                # Apply feature selection
                if self.selected_features:
                    X_train_processed = X_train_processed[self.selected_features]
                    X_val_processed = X_val_processed[self.selected_features]
            else:
                X_train_processed = X_train_aug[self.selected_features] if self.selected_features else X_train_aug
                X_val_processed = X_val_iter[self.selected_features] if self.selected_features else X_val_iter

            # Train model
            model_iter = clone(self.base_model)
            model_iter.fit(X_train_processed, y_train_aug)

            # Make predictions
            y_pred_iter = model_iter.predict(X_val_processed)

            # Store predictions
            for idx, pred in zip(remain_indices, y_pred_iter):
                self.sample_predictions[idx].append(pred)

            # Calculate iteration metrics
            iter_r2 = r2_score(y_val_iter, y_pred_iter)
            iter_rmse = np.sqrt(mean_squared_error(y_val_iter, y_pred_iter))
            iter_mae = mean_absolute_error(y_val_iter, y_pred_iter)

            # Store iteration results
            iteration_result = {
                'iteration': iteration + 1,
                'moved_samples': len(move_indices),
                'validation_samples': len(remain_indices),
                'r2': iter_r2,
                'rmse': iter_rmse,
                'mae': iter_mae,
                'move_indices': move_indices.tolist(),
                'remain_indices': remain_indices
            }
            self.iteration_results.append(iteration_result)

            # Track convergence
            if iteration >= 9:  # Need at least 10 iterations for running average
                recent_r2 = [result['r2'] for result in self.iteration_results[-10:]]
                running_mean = np.mean(recent_r2)
                running_std = np.std(recent_r2)

                self.convergence_metrics.append({
                    'iteration': iteration + 1,
                    'running_r2_mean': running_mean,
                    'running_r2_std': running_std,
                    'current_r2': iter_r2
                })

        logger.info(f"Completed {required_iterations} iterations")

        # Calculate final aggregated results
        final_results = self._calculate_final_results(X_validation, y_validation)

        # Create comprehensive visualizations
        self._create_comprehensive_visualizations(final_results)

        # Save results
        self._save_results(final_results)

        logger.info("=" * 60)
        logger.info("STRATIFIED VALIDATION RESAMPLING COMPLETED")
        logger.info(f"Final R²: {final_results['aggregated_metrics']['r2']:.4f}")
        logger.info(
            f"Sample usage: {final_results['sample_usage_stats']['mean_usage']:.1f} ± {final_results['sample_usage_stats']['std_usage']:.1f}")
        logger.info("=" * 60)

        return final_results

    def _calculate_final_results(self, X_validation: pd.DataFrame, y_validation: pd.Series) -> Dict[str, Any]:
        """Calculate comprehensive final results from all iterations."""
        logger.info("Calculating final aggregated results...")

        # Calculate mean predictions for each sample
        mean_predictions = {}
        prediction_std = {}
        prediction_counts = {}

        for idx in X_validation.index:
            if self.sample_predictions[idx]:
                predictions = np.array(self.sample_predictions[idx])
                mean_predictions[idx] = np.mean(predictions)
                prediction_std[idx] = np.std(predictions)
                prediction_counts[idx] = len(predictions)
            else:
                mean_predictions[idx] = np.nan
                prediction_std[idx] = np.nan
                prediction_counts[idx] = 0

        # Create final prediction arrays - ensure they are numpy arrays
        y_true_final = np.array(y_validation.values)
        y_pred_final = np.array([mean_predictions[idx] for idx in X_validation.index])
        y_pred_std_final = np.array([prediction_std[idx] for idx in X_validation.index])

        # Remove any NaN predictions for metric calculations
        valid_mask = ~(np.isnan(y_pred_final) | np.isnan(y_true_final))

        if np.sum(valid_mask) == 0:
            logger.error("No valid predictions found!")
            raise ValueError("All predictions are NaN")

        y_true_valid = y_true_final[valid_mask]
        y_pred_valid = y_pred_final[valid_mask]

        # Calculate aggregated metrics on valid predictions only
        aggregated_r2 = r2_score(y_true_valid, y_pred_valid)
        aggregated_rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
        aggregated_mae = mean_absolute_error(y_true_valid, y_pred_valid)

        # Per-iteration metrics
        iteration_r2_scores = [result['r2'] for result in self.iteration_results]
        iteration_rmse_scores = [result['rmse'] for result in self.iteration_results]

        # Sample usage statistics
        usage_counts = list(self.sample_usage_count.values())
        prediction_counts_values = list(prediction_counts.values())

        # Bias analysis by target range
        bias_analysis = self._analyze_prediction_bias(y_true_final, y_pred_final, y_pred_std_final)

        # Convergence analysis
        convergence_analysis = self._analyze_convergence()

        logger.info(f"Final aggregated metrics calculated:")
        logger.info(f"  Valid predictions: {np.sum(valid_mask)}/{len(y_true_final)}")
        logger.info(f"  R²: {aggregated_r2:.4f}")
        logger.info(f"  RMSE: {aggregated_rmse:.4f}")
        logger.info(f"  MAE: {aggregated_mae:.4f}")

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
            'predictions': {
                'y_true': y_true_final,  # Keep as numpy array
                'y_pred_mean': y_pred_final,  # Keep as numpy array
                'y_pred_std': y_pred_std_final,  # Keep as numpy array
                'sample_indices': X_validation.index.tolist(),
                'valid_mask': valid_mask
            },
            'bias_analysis': bias_analysis,
            'convergence_analysis': convergence_analysis,
            'experiment_config': {
                'move_percentage': self.move_percentage,
                'total_iterations': len(self.iteration_results),
                'min_sample_usage': self.min_sample_usage,
                'random_state': self.random_state
            }
        }

    def _analyze_prediction_bias(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_std: np.ndarray) -> Dict[
        str, Any]:
        """Analyze prediction bias across different target value ranges."""
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

    def _analyze_convergence(self) -> Dict[str, Any]:
        """Analyze convergence of the resampling process."""
        if not self.convergence_metrics:
            return {'converged': False, 'reason': 'Insufficient iterations for convergence analysis'}

        # Extract convergence data
        iterations = [m['iteration'] for m in self.convergence_metrics]
        running_means = [m['running_r2_mean'] for m in self.convergence_metrics]
        running_stds = [m['running_r2_std'] for m in self.convergence_metrics]

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
                'iterations_analyzed': len(self.convergence_metrics)
            }

        return {'converged': False, 'reason': 'Insufficient iterations for convergence analysis'}

    def _create_comprehensive_visualizations(self, results: Dict[str, Any]) -> None:
        """Create comprehensive visualizations of the resampling results."""
        logger.info("Creating comprehensive visualizations...")

        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")

        # Create main results figure
        fig = plt.figure(figsize=(20, 16))

        # 1. Prediction Accuracy Plot (Top Left)
        ax1 = plt.subplot(3, 3, 1)
        y_true = results['predictions']['y_true']
        y_pred = results['predictions']['y_pred_mean']
        y_pred_std = results['predictions']['y_pred_std']

        scatter = ax1.scatter(y_true, y_pred, c=y_pred_std, cmap='viridis', alpha=0.7, s=50)

        # Perfect prediction line
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)

        ax1.set_xlabel('True Values', fontsize=12)
        ax1.set_ylabel('Mean Predicted Values', fontsize=12)
        ax1.set_title(f'Prediction Accuracy\nR² = {results["aggregated_metrics"]["r2"]:.4f}', fontsize=14)
        ax1.grid(True, alpha=0.3)

        # Add colorbar for uncertainty
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Prediction Std', fontsize=10)

        # 2. Convergence Plot (Top Center)
        ax2 = plt.subplot(3, 3, 2)
        if self.convergence_metrics and len(self.convergence_metrics) > 0:
            iterations = [m['iteration'] for m in self.convergence_metrics]
            running_means = [m['running_r2_mean'] for m in self.convergence_metrics]
            running_stds = [m['running_r2_std'] for m in self.convergence_metrics]

            ax2.plot(iterations, running_means, 'b-', linewidth=2, label='Running Mean R²')
            ax2.fill_between(iterations,
                             np.array(running_means) - np.array(running_stds),
                             np.array(running_means) + np.array(running_stds),
                             alpha=0.3, color='blue')

            ax2.set_xlabel('Iteration', fontsize=12)
            ax2.set_ylabel('Running R² (10-iter window)', fontsize=12)
            ax2.set_title('Convergence Analysis', fontsize=14)
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            # Add convergence indicator
            if results['convergence_analysis']['converged']:
                ax2.text(0.05, 0.95, 'CONVERGED ✓', transform=ax2.transAxes,
                         fontsize=12, color='green', weight='bold',
                         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        else:
            ax2.text(0.5, 0.5, 'Not enough iterations\nfor convergence analysis\n(need 10+)',
                     transform=ax2.transAxes, ha='center', va='center', fontsize=12)
            ax2.set_title('Convergence Analysis', fontsize=14)
            ax2.axis('off')

        # 3. Sample Usage Distribution (Top Right)
        ax3 = plt.subplot(3, 3, 3)
        usage_counts = list(self.sample_usage_count.values())

        # SAFETY CHECK: Only create histogram if we have enough data
        if len(usage_counts) > 0:
            ax3.hist(usage_counts, bins=min(15, max(3, len(usage_counts) // 2)),
                     alpha=0.7, color='skyblue', edgecolor='black')
            ax3.axvline(self.min_sample_usage, color='red', linestyle='--',
                        label=f'Target: {self.min_sample_usage}')
            ax3.axvline(np.mean(usage_counts), color='green', linestyle='-',
                        label=f'Mean: {np.mean(usage_counts):.1f}')
            ax3.set_xlabel('Times Used in Training', fontsize=12)
            ax3.set_ylabel('Number of Samples', fontsize=12)
            ax3.set_title('Sample Usage Distribution', fontsize=14)
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'No usage data available',
                     transform=ax3.transAxes, ha='center', va='center', fontsize=12)
            ax3.axis('off')

        # 4. Iteration R² Distribution (Middle Left)
        ax4 = plt.subplot(3, 3, 4)
        iteration_r2s = [result['r2'] for result in self.iteration_results]

        # SAFETY CHECK: Filter out NaN values and check if we have valid data
        iteration_r2s_valid = [r2 for r2 in iteration_r2s if not np.isnan(r2) and np.isfinite(r2)]

        if len(iteration_r2s_valid) > 1:
            # Adaptive bin count based on data size
            n_bins = min(20, max(3, len(iteration_r2s_valid) // 3))
            ax4.hist(iteration_r2s_valid, bins=n_bins, alpha=0.7, color='lightcoral', edgecolor='black')
            ax4.axvline(results['aggregated_metrics']['r2'], color='red', linestyle='-',
                        label=f'Final R²: {results["aggregated_metrics"]["r2"]:.4f}')
            ax4.axvline(np.mean(iteration_r2s_valid), color='blue', linestyle='--',
                        label=f'Mean: {np.mean(iteration_r2s_valid):.4f}')
            ax4.set_xlabel('R² Score', fontsize=12)
            ax4.set_ylabel('Frequency', fontsize=12)
            ax4.set_title('Per-Iteration R² Distribution', fontsize=14)
            ax4.grid(True, alpha=0.3)
            ax4.legend()
        else:
            # Not enough valid data for histogram
            ax4.text(0.5, 0.5, f'Too few valid R² values\nfor histogram\n({len(iteration_r2s_valid)} values)',
                     transform=ax4.transAxes, ha='center', va='center', fontsize=12,
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
            ax4.axis('off')

        # 5. Bias Analysis by Target Range (Middle Center)
        ax5 = plt.subplot(3, 3, 5)
        bias_data = results['bias_analysis']
        if bias_data and len(bias_data) > 0:
            ranges = list(bias_data.keys())
            biases = [bias_data[r]['bias'] for r in ranges]
            n_samples = [bias_data[r]['n_samples'] for r in ranges]

            colors = ['red' if b > 0 else 'blue' for b in biases]
            bars = ax5.bar(range(len(ranges)), biases, color=colors, alpha=0.7)

            # Add sample counts on bars
            for i, (bar, n) in enumerate(zip(bars, n_samples)):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width() / 2, height + 0.01 * np.sign(height) if height != 0 else 0.01,
                         f'n={n}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)

            ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax5.set_xlabel('Target Value Range', fontsize=12)
            ax5.set_ylabel('Mean Bias (Pred - True)', fontsize=12)
            ax5.set_title('Prediction Bias by Range', fontsize=14)
            ax5.set_xticks(range(len(ranges)))
            ax5.set_xticklabels(ranges, rotation=45, ha='right')
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'Insufficient data\nfor bias analysis',
                     transform=ax5.transAxes, ha='center', va='center', fontsize=12)
            ax5.axis('off')

        # 6. Uncertainty Analysis (Middle Right)
        ax6 = plt.subplot(3, 3, 6)
        uncertainties = results['predictions']['y_pred_std']
        if len(y_true) > 0:
            ax6.scatter(y_true, uncertainties, alpha=0.7, s=30, c='purple')
            ax6.set_xlabel('True Values', fontsize=12)
            ax6.set_ylabel('Prediction Uncertainty (Std)', fontsize=12)
            ax6.set_title('Prediction Uncertainty vs True Values', fontsize=14)
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'No uncertainty data',
                     transform=ax6.transAxes, ha='center', va='center', fontsize=12)
            ax6.axis('off')

        # 7. Residuals Analysis (Bottom Left)
        ax7 = plt.subplot(3, 3, 7)
        if len(y_true) > 0 and len(y_pred) > 0:
            residuals = y_pred - y_true
            ax7.scatter(y_pred, residuals, alpha=0.7, s=30, c='orange')
            ax7.axhline(y=0, color='red', linestyle='--', alpha=0.8)
            ax7.set_xlabel('Predicted Values', fontsize=12)
            ax7.set_ylabel('Residuals (Pred - True)', fontsize=12)
            ax7.set_title(f'Residuals Analysis\nRMSE = {results["aggregated_metrics"]["rmse"]:.4f}', fontsize=14)
            ax7.grid(True, alpha=0.3)
        else:
            ax7.text(0.5, 0.5, 'No residual data',
                     transform=ax7.transAxes, ha='center', va='center', fontsize=12)
            ax7.axis('off')

        # 8. Performance Summary (Bottom Center)
        ax8 = plt.subplot(3, 3, 8)
        ax8.axis('off')

        # Add warning for small dataset
        warning_text = ""
        if len(y_true) < 10:
            warning_text = "\n⚠️ SMALL DATASET WARNING\nResults may be unreliable\n\n"

        summary_text = warning_text + f"""
    STRATIFIED VALIDATION RESULTS

    Final Metrics:
    - R² Score: {results['aggregated_metrics']['r2']:.4f}
    - RMSE: {results['aggregated_metrics']['rmse']:.4f}
    - MAE: {results['aggregated_metrics']['mae']:.4f}

    Iteration Stats:
    - Total Iterations: {len(self.iteration_results)}
    - Mean R²: {results['iteration_metrics']['mean_r2']:.4f} ± {results['iteration_metrics']['std_r2']:.4f}
    - Best R²: {results['iteration_metrics']['best_r2']:.4f}

    Sample Usage:
    - Mean Usage: {results['sample_usage_stats']['mean_usage']:.1f} ± {results['sample_usage_stats']['std_usage']:.1f}
    - Min Usage: {results['sample_usage_stats']['min_usage']}
    - Max Usage: {results['sample_usage_stats']['max_usage']}

    Convergence: {'✓ CONVERGED' if results['convergence_analysis']['converged'] else '✗ NOT CONVERGED'}
        """

        bg_color = 'lightyellow' if len(y_true) < 10 else 'lightgray'
        ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.8))

        # 9. Model Comparison (Bottom Right)
        ax9 = plt.subplot(3, 3, 9)

        # Create a simple comparison showing improvement over baseline
        baseline_r2 = 0.0  # Assuming baseline is predicting mean
        improvement = results['aggregated_metrics']['r2'] - baseline_r2

        categories = ['Baseline\n(Mean)', 'Stratified\nResampling']
        scores = [baseline_r2, results['aggregated_metrics']['r2']]
        colors = ['lightgray', 'green']

        bars = ax9.bar(categories, scores, color=colors, alpha=0.7)
        ax9.set_ylabel('R² Score', fontsize=12)
        ax9.set_title(f'Method Comparison\n(Improvement: +{improvement:.4f})', fontsize=14)
        ax9.set_ylim(0, 1)
        ax9.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax9.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                     f'{score:.4f}', ha='center', va='bottom', fontsize=12, weight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(self.experiment_dir, 'comprehensive_validation_analysis.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # Create additional detailed plots only if we have enough data
        if len(self.iteration_results) > 10:
            self._create_detailed_plots(results)
        else:
            logger.info(f"Skipping detailed plots due to small dataset ({len(self.iteration_results)} iterations)")

        logger.info(f"Visualizations saved to {self.experiment_dir}")

    def _create_detailed_plots(self, results: Dict[str, Any]) -> None:
        """Create additional detailed plots."""

        # 1. Iteration-by-iteration performance
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        iterations = [r['iteration'] for r in self.iteration_results]
        r2_scores = [r['r2'] for r in self.iteration_results]
        rmse_scores = [r['rmse'] for r in self.iteration_results]

        ax1.plot(iterations, r2_scores, 'b-', alpha=0.7, linewidth=1)
        ax1.axhline(y=results['aggregated_metrics']['r2'], color='red', linestyle='--',
                    label=f'Final R²: {results["aggregated_metrics"]["r2"]:.4f}')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('R² Score')
        ax1.set_title('R² Score by Iteration')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.plot(iterations, rmse_scores, 'g-', alpha=0.7, linewidth=1)
        ax2.axhline(y=results['aggregated_metrics']['rmse'], color='red', linestyle='--',
                    label=f'Final RMSE: {results["aggregated_metrics"]["rmse"]:.4f}')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('RMSE')
        ax2.set_title('RMSE by Iteration')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.experiment_dir, 'iteration_performance.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Sample-specific prediction analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot prediction uncertainty vs true values
        y_true = results['predictions']['y_true']
        y_pred_std = results['predictions']['y_pred_std']

        ax1.scatter(y_true, y_pred_std, alpha=0.7, s=50, c='purple')
        ax1.set_xlabel('True Values')
        ax1.set_ylabel('Prediction Uncertainty (Std)')
        ax1.set_title('Prediction Uncertainty Analysis')
        ax1.grid(True, alpha=0.3)

        # Plot sample usage vs prediction accuracy
        sample_indices = results['predictions']['sample_indices']
        usage_counts = [self.sample_usage_count[idx] for idx in sample_indices]
        absolute_errors = np.abs(results['predictions']['y_pred_mean'] - results['predictions']['y_true'])

        ax2.scatter(usage_counts, absolute_errors, alpha=0.7, s=50, c='orange')
        ax2.set_xlabel('Sample Usage Count')
        ax2.set_ylabel('Absolute Error')
        ax2.set_title('Sample Usage vs Prediction Error')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.experiment_dir, 'sample_analysis.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save comprehensive results to files."""
        logger.info("Saving results...")

        # Convert predictions to numpy arrays for calculations
        y_true_array = np.array(results['predictions']['y_true'])
        y_pred_mean_array = np.array(results['predictions']['y_pred_mean'])
        y_pred_std_array = np.array(results['predictions']['y_pred_std'])

        # Calculate errors using numpy arrays
        absolute_error = np.abs(y_pred_mean_array - y_true_array)
        relative_error = np.abs(y_pred_mean_array - y_true_array) / np.abs(y_true_array)

        # Replace inf values in relative error with NaN
        relative_error = np.where(np.isinf(relative_error), np.nan, relative_error)

        # Save main results as JSON
        results_copy = results.copy()

        # Convert numpy arrays to lists for JSON serialization
        predictions = results_copy['predictions']
        predictions['y_true'] = y_true_array.tolist()
        predictions['y_pred_mean'] = y_pred_mean_array.tolist()
        predictions['y_pred_std'] = y_pred_std_array.tolist()

        with open(os.path.join(self.experiment_dir, 'stratified_validation_results.json'), 'w') as f:
            json.dump(results_copy, f, indent=4, default=str)

        # Save detailed predictions as CSV
        pred_df = pd.DataFrame({
            'sample_index': results['predictions']['sample_indices'],
            'y_true': y_true_array,
            'y_pred_mean': y_pred_mean_array,
            'y_pred_std': y_pred_std_array,
            'usage_count': [self.sample_usage_count[idx] for idx in results['predictions']['sample_indices']],
            'absolute_error': absolute_error,
            'relative_error': relative_error
        })
        pred_df.to_csv(os.path.join(self.experiment_dir, 'detailed_predictions.csv'), index=False)

        # Save iteration results
        iter_df = pd.DataFrame(self.iteration_results)
        iter_df.to_csv(os.path.join(self.experiment_dir, 'iteration_results.csv'), index=False)

        # Save convergence metrics if available
        if self.convergence_metrics:
            conv_df = pd.DataFrame(self.convergence_metrics)
            conv_df.to_csv(os.path.join(self.experiment_dir, 'convergence_metrics.csv'), index=False)

        # Save bias analysis
        if results['bias_analysis']:
            bias_df = pd.DataFrame.from_dict(results['bias_analysis'], orient='index')
            bias_df.to_csv(os.path.join(self.experiment_dir, 'bias_analysis.csv'))

        # Save summary report
        self._create_summary_report(results)

        logger.info(f"All results saved to {self.experiment_dir}")

    def _create_summary_report(self, results: Dict[str, Any]) -> None:
        """Create a comprehensive summary report."""

        report = f"""
STRATIFIED VALIDATION RESAMPLING REPORT
=======================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXPERIMENT CONFIGURATION
------------------------
• Move Percentage: {self.move_percentage * 100:.1f}%
• Total Iterations: {len(self.iteration_results)}
• Minimum Sample Usage: {self.min_sample_usage}
• Random State: {self.random_state}

FINAL PERFORMANCE METRICS
-------------------------
• R² Score: {results['aggregated_metrics']['r2']:.6f}
• RMSE: {results['aggregated_metrics']['rmse']:.6f}
• MAE: {results['aggregated_metrics']['mae']:.6f}
• Samples: {results['aggregated_metrics']['samples']}

ITERATION STATISTICS
-------------------
• Mean R²: {results['iteration_metrics']['mean_r2']:.6f} ± {results['iteration_metrics']['std_r2']:.6f}
• Mean RMSE: {results['iteration_metrics']['mean_rmse']:.6f} ± {results['iteration_metrics']['std_rmse']:.6f}
• Best R²: {results['iteration_metrics']['best_r2']:.6f}
• Worst R²: {results['iteration_metrics']['worst_r2']:.6f}

SAMPLE USAGE ANALYSIS
--------------------
• Mean Usage: {results['sample_usage_stats']['mean_usage']:.2f} ± {results['sample_usage_stats']['std_usage']:.2f}
• Usage Range: {results['sample_usage_stats']['min_usage']} - {results['sample_usage_stats']['max_usage']}
• Mean Predictions per Sample: {results['sample_usage_stats']['mean_predictions']:.2f}

CONVERGENCE ANALYSIS
-------------------
• Status: {'CONVERGED' if results['convergence_analysis']['converged'] else 'NOT CONVERGED'}
"""

        if results['convergence_analysis']['converged']:
            report += f"""• Final Running Mean: {results['convergence_analysis']['final_mean']:.6f}
• Final Running Std: {results['convergence_analysis']['final_std']:.6f}
• Stability Metric: {results['convergence_analysis']['stability_metric']:.6f}
"""

        report += "\nBIAS ANALYSIS BY TARGET RANGE\n"
        report += "-----------------------------\n"

        if results['bias_analysis']:
            for range_name, bias_data in results['bias_analysis'].items():
                report += f"• {range_name}:\n"
                report += f"  - Samples: {bias_data['n_samples']}\n"
                report += f"  - Bias: {bias_data['bias']:.6f}\n"
                report += f"  - Absolute Bias: {bias_data['abs_bias']:.6f}\n"
                r2_value = bias_data['r2']
                if np.isnan(r2_value):
                    r2_str = 'N/A'
                else:
                    r2_str = f"{r2_value:.6f}"

                report += f"  - R²: {r2_str}\n"
                report += f"  - Mean Uncertainty: {bias_data['mean_uncertainty']:.6f}\n\n"

        report += "\nRECOMMENDations\n"
        report += "---------------\n"

        # Generate recommendations based on results
        recommendations = []

        if results['aggregated_metrics']['r2'] > 0.8:
            recommendations.append("✓ Excellent model performance achieved")
        elif results['aggregated_metrics']['r2'] > 0.6:
            recommendations.append("+ Good model performance, consider minor improvements")
        else:
            recommendations.append("! Model performance needs improvement")

        if results['convergence_analysis']['converged']:
            recommendations.append("✓ Process converged successfully")
        else:
            recommendations.append("! Consider increasing iterations for better convergence")

        if results['sample_usage_stats']['min_usage'] >= self.min_sample_usage:
            recommendations.append("✓ All samples used sufficiently")
        else:
            recommendations.append("! Some samples underused, consider more iterations")

        # Check for bias issues
        if results['bias_analysis']:
            max_bias = max(abs(bias_data['bias']) for bias_data in results['bias_analysis'].values())
            if max_bias > 0.1:
                recommendations.append("! Significant bias detected in some ranges")
            else:
                recommendations.append("✓ Bias levels acceptable across ranges")

        for rec in recommendations:
            report += rec + "\n"

        # Save report
        with open(os.path.join(self.experiment_dir, 'summary_report.txt'), 'w', encoding='utf-8') as f:
            f.write(report)

    def get_best_predictions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the final aggregated predictions.

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (mean_predictions, prediction_std, sample_indices)
        """
        if not self.sample_predictions:
            raise ValueError("No predictions available. Run validation resampling first.")

        mean_preds = []
        std_preds = []
        indices = []

        for idx, predictions in self.sample_predictions.items():
            if predictions:
                mean_preds.append(np.mean(predictions))
                std_preds.append(np.std(predictions))
                indices.append(idx)

        return np.array(mean_preds), np.array(std_preds), np.array(indices)


def run_stratified_validation_analysis(X_train: pd.DataFrame,
                                       y_train: pd.Series,
                                       X_validation: pd.DataFrame,
                                       y_validation: pd.Series,
                                       model_template: Any,
                                       preprocessor: Any = None,
                                       selected_features: List[str] = None,
                                       move_percentage: float = 0.7,
                                       min_iterations: int = 100,
                                       min_sample_usage: int = 5,
                                       random_state: int = 42,
                                       experiment_dir: str = None) -> Dict[str, Any]:
    """
    Convenience function to run stratified validation analysis.

    This function provides a simple interface to the StratifiedValidationResampler
    for your specific use case.

    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training targets
    X_validation : pd.DataFrame
        Validation features (the problematic validation set)
    y_validation : pd.Series
        Validation targets
    model_template : sklearn estimator
        Model to use for training
    preprocessor : optional
        Preprocessing pipeline
    selected_features : List[str], optional
        Features to use for modeling
    move_percentage : float, default=0.7
        Percentage of validation samples to move to training each iteration
    min_iterations : int, default=100
        Minimum number of iterations
    min_sample_usage : int, default=5
        Minimum times each validation sample should be used
    random_state : int, default=42
        Random seed
    experiment_dir : str, optional
        Directory for results

    Returns:
    --------
    Dict[str, Any]
        Comprehensive results dictionary
    """
    logger.info("=" * 60)
    logger.info("STARTING STRATIFIED VALIDATION ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_validation)}")
    logger.info(f"Move percentage: {move_percentage * 100:.1f}%")
    logger.info(f"Target iterations: {min_iterations}+")
    logger.info(f"Target sample usage: {min_sample_usage}+")

    # Initialize resampler
    resampler = StratifiedValidationResampler(
        move_percentage=move_percentage,
        min_iterations=min_iterations,
        min_sample_usage=min_sample_usage,
        random_state=random_state,
        experiment_dir=experiment_dir or f'stratified_validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )

    # Run the analysis
    results = resampler.run_validation_resampling(
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        model_template=model_template,
        preprocessor=preprocessor,
        selected_features=selected_features
    )

    logger.info("=" * 60)
    logger.info("STRATIFIED VALIDATION ANALYSIS COMPLETED")
    logger.info(f"Final R²: {results['aggregated_metrics']['r2']:.4f}")
    logger.info(f"Results directory: {resampler.experiment_dir}")
    logger.info("=" * 60)

    return results


# Integration utilities for main.py
def integrate_stratified_validation_in_retrain_function(retrain_results: Dict[str, Any],
                                                        model_name: str,
                                                        experiment_dir: str) -> Dict[str, Any]:
    """
    Integration function to add stratified validation to existing retrain results.

    This function takes your existing retrain results and adds stratified validation
    analysis using the validation dataset.

    Parameters:
    -----------
    retrain_results : Dict[str, Any]
        Results from retrain_best_model_with_gap_optimized_cv
    model_name : str
        Name of the model used
    experiment_dir : str
        Main experiment directory

    Returns:
    --------
    Dict[str, Any]
        Enhanced results with stratified validation analysis
    """
    logger.info("Integrating stratified validation analysis...")

    # Extract components from retrain results
    datasets = retrain_results['datasets']
    retrained_model = retrain_results['retrained_model']

    X_train = datasets['X_train']
    y_train = datasets['y_train']
    X_validation = datasets['X_validation']
    y_validation = datasets['y_validation']

    # Create stratified validation directory
    stratified_dir = os.path.join(experiment_dir, 'stratified_validation_analysis')

    # Run stratified validation analysis
    stratified_results = run_stratified_validation_analysis(
        move_percentage=0.90,
        min_iterations=200,
        min_sample_usage=8,
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        model_template=retrained_model,
        selected_features=list(X_train.columns),
        experiment_dir=stratified_dir
    )

    # Compare with original validation results
    original_val_r2 = retrain_results['metrics']['validation']['r2']
    stratified_r2 = stratified_results['aggregated_metrics']['r2']
    improvement = stratified_r2 - original_val_r2

    logger.info(f"Validation Performance Comparison:")
    logger.info(f"  Original Validation R²: {original_val_r2:.4f}")
    logger.info(f"  Stratified Validation R²: {stratified_r2:.4f}")
    logger.info(f"  Improvement: {improvement:+.4f}")

    # Add stratified results to retrain results
    retrain_results['stratified_validation'] = {
        'results': stratified_results,
        'improvement': improvement,
        'original_val_r2': original_val_r2,
        'stratified_val_r2': stratified_r2,
        'analysis_dir': stratified_dir
    }

    return retrain_results

def retrain_best_model_with_gap_optimized_cv(best_results, best_model_name, experiment_dir,
                                             original_data, target_column='Bioaccessible Pb',
                                             validation_indices=None, validation_size=0.15,
                                             test_size=0.15, random_state=42,
                                             custom_validation_criteria=None,
                                             use_gap_optimization=True):
    """
    Retrain the best performing model using the same gap-optimized CV approach as the main pipeline.

    Parameters:
    -----------
    best_results : dict
        Results dictionary from the best performing pipeline
    best_model_name : str
        Name of the best performing model
    experiment_dir : str
        Main experiment directory
    original_data : DataFrame
        Original complete dataset before any splits
    target_column : str, default='Bioaccessible Pb'
        Name of the target column
    validation_indices : list or array-like, optional
        Specific row indices to use for validation set
    validation_size : float, default=0.15
        Proportion of data to use for validation (if validation_indices not provided)
    test_size : float, default=0.15
        Proportion of data to use for testing
    random_state : int, default=42
        Random seed for reproducible splits
    custom_validation_criteria : dict, optional
        Custom criteria for selecting validation data
    use_gap_optimization : bool, default=True
        Whether to use gap-optimized cross-validation approach

    Returns:
    --------
    dict
        Results dictionary with retrained model and validation metrics
    """
    logger.info("=" * 60)
    logger.info("ROTATE RETRAINING WITH GAP-OPTIMIZED CROSS-VALIDATION")
    logger.info("=" * 60)

    # Create retraining directory
    retrain_dir = os.path.join(experiment_dir, 'retrained_best_model_gap_optimized')
    os.makedirs(retrain_dir, exist_ok=True)

    # Extract features and target
    X_full = original_data.drop(target_column, axis=1)
    y_full = original_data[target_column]

    logger.info(f"DATA Full dataset: {X_full.shape[0]} samples, {X_full.shape[1]} features")

    # STEP 1: Create custom validation split (same as before)
    logger.info("TARGET STEP 1: Creating custom data splits...")

    if validation_indices is not None:
        validation_indices = np.array(validation_indices)
        remaining_indices = np.setdiff1d(np.arange(len(X_full)), validation_indices)

        X_validation = X_full.iloc[validation_indices]
        y_validation = y_full.iloc[validation_indices]
        X_remaining = X_full.iloc[remaining_indices]
        y_remaining = y_full.iloc[remaining_indices]

        logger.info(f"SUCCESS Using provided validation indices: {len(validation_indices)} samples")

    elif custom_validation_criteria is not None:
        validation_mask = create_validation_mask(X_full, y_full, custom_validation_criteria)
        validation_indices = np.where(validation_mask)[0]
        remaining_indices = np.where(~validation_mask)[0]

        X_validation = X_full.iloc[validation_indices]
        y_validation = y_full.iloc[validation_indices]
        X_remaining = X_full.iloc[remaining_indices]
        y_remaining = y_full.iloc[remaining_indices]

        logger.info(f"SUCCESS Custom validation criteria selected: {len(validation_indices)} samples")
        logger.info(f"   Criteria: {custom_validation_criteria}")

    else:
        # First split: separate validation set
        X_remaining, X_validation, y_remaining, y_validation = train_test_split(
            X_full, y_full, test_size=validation_size, random_state=random_state, stratify=None
        )
        validation_indices = X_validation.index.values

        logger.info(f"SUCCESS Random validation split: {len(X_validation)} samples ({validation_size * 100:.1f}%)")

    # Split remaining data into train/test
    remaining_test_size = test_size / (1 - validation_size)
    X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
        X_remaining, y_remaining, test_size=remaining_test_size, random_state=random_state
    )

    logger.info(f"DATA Final splits:")
    logger.info(f"   Training: {len(X_train_new)} samples ({len(X_train_new) / len(X_full) * 100:.1f}%)")
    logger.info(f"   Testing: {len(X_test_new)} samples ({len(X_test_new) / len(X_full) * 100:.1f}%)")
    logger.info(f"   Validation: {len(X_validation)} samples ({len(X_validation) / len(X_full) * 100:.1f}%)")

    # STEP 2: Apply the same preprocessing pipeline (same as before)
    logger.info("ROTATE STEP 2: Applying preprocessing pipeline...")

    preprocessor = best_results['preprocessor']

    X_train_processed = preprocessor.transform(X_train_new)
    X_test_processed = preprocessor.transform(X_test_new)
    X_validation_processed = preprocessor.transform(X_validation)

    y_train_processed = preprocessor.transform_target(y_train_new)
    y_test_processed = preprocessor.transform_target(y_test_new)
    y_validation_processed = preprocessor.transform_target(y_validation)

    logger.info(f"SUCCESS Preprocessing completed")
    logger.info(f"   Processed features: {X_train_processed.shape[1]}")

    main_X = pd.concat([X_train_processed, X_test_processed])
    main_y = pd.concat([y_train_processed, y_test_processed])

    # STEP 3: Apply feature selection (same as before)
    logger.info("TARGET STEP 3: Applying feature selection...")

    selected_features = best_results.get('selected_features', best_results['X_train_selected'].columns.tolist())

    X_train_selected = X_train_processed[selected_features]
    X_test_selected = X_test_processed[selected_features]
    X_validation_selected = X_validation_processed[selected_features]

    logger.info(f"SUCCESS Selected {len(selected_features)} features")

    # STEP 4: Enhanced Model Retraining with Gap-Optimized CV
    logger.info("STARTING STEP 4: Enhanced Model Retraining with Gap-Optimized CV...")

    if use_gap_optimization:
        # Use the same gap-optimized approach as the main pipeline
        from cross_validation import GapOptimizedCrossValidator, get_optimal_cv_params

        # Get optimal CV parameters for this model type
        cv_params = get_optimal_cv_params(best_model_name)

        # Initialize gap-optimized cross-validator
        gap_cv = GapOptimizedCrossValidator(
            n_splits=cv_params['n_splits'],
            n_repeats=cv_params['n_repeats'],
            random_state=random_state,
            experiment_dir=retrain_dir
        )
        gap_cv.gap_threshold = cv_params['gap_threshold']

        logger.info(f"TARGET Using gap-optimized CV with threshold: {cv_params['gap_threshold']}")
        logger.info(f"   CV folds: {cv_params['n_splits']}, repeats: {cv_params['n_repeats']}")

        # Get the best model configuration
        best_model = best_results['models'][best_model_name]

        # Clone and evaluate with gap-optimized CV
        from sklearn.base import clone
        retrained_model = clone(best_model)

        # Perform gap-optimized cross-validation on training data
        cv_result = gap_cv.evaluate_model(retrained_model, X_train_selected, y_train_processed, best_model_name)

        logger.info(f"DATA Gap-optimized CV results:")
        logger.info(f"   Train R² = {cv_result['train_r2_mean']:.4f} +/- {cv_result['train_r2_std']:.4f}")
        logger.info(f"   Test R² = {cv_result['test_r2_mean']:.4f} +/- {cv_result['test_r2_std']:.4f}")
        logger.info(f"   Gap = {cv_result['r2_gap']:.4f}")
        logger.info(f"   Used {cv_result['used_folds']}/{cv_result['total_folds']} folds")

        # Check if gap is acceptable
        if cv_result['r2_gap'] <= cv_params['gap_threshold']:
            logger.info(
                f"SUCCESS Gap {cv_result['r2_gap']:.4f} <= threshold {cv_params['gap_threshold']:.4f} - Model is well-regularized")
        else:
            logger.warning(
                f"WARNING Gap {cv_result['r2_gap']:.4f} > threshold {cv_params['gap_threshold']:.4f} - Potential overfitting detected")

        # Final training on all training data
        logger.info("TARGET Final training on all training data...")
        retrained_model.fit(X_train_selected, y_train_processed)

    else:
        logger.info("ROTATE Using standard retraining without gap optimization...")

        # Standard approach (original method)
        best_model = best_results['models'][best_model_name]
        from sklearn.base import clone
        retrained_model = clone(best_model)
        retrained_model.fit(X_train_selected, y_train_processed)

        # Simple cross-validation for comparison
        from sklearn.model_selection import cross_val_score
        cv_scores = cross_val_score(retrained_model, X_train_selected, y_train_processed,
                                    cv=5, scoring='r2', n_jobs=-1)

        cv_result = {
            'train_r2_mean': np.nan,  # Not available in simple CV
            'train_r2_std': np.nan,
            'test_r2_mean': cv_scores.mean(),
            'test_r2_std': cv_scores.std(),
            'r2_gap': np.nan,
            'used_folds': len(cv_scores),
            'total_folds': len(cv_scores)
        }

    logger.info(f"SUCCESS Model {best_model_name} retrained successfully")

    # STEP 5: Comprehensive evaluation on all splits
    logger.info("DATA STEP 5: Comprehensive evaluation on all splits...")

    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    # Predictions on all sets
    y_train_pred = retrained_model.predict(X_train_selected)
    y_test_pred = retrained_model.predict(X_test_selected)
    y_validation_pred = retrained_model.predict(X_validation_selected)

    # Calculate metrics for all sets
    metrics = {}

    for set_name, y_true, y_pred in [
        ('train', y_train_processed, y_train_pred),
        ('test', y_test_processed, y_test_pred),
        ('validation', y_validation_processed, y_validation_pred)
    ]:
        metrics[set_name] = {
            'r2': r2_score(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'samples': len(y_true)
        }

    # Log results
    logger.info("RESULTS ENHANCED RETRAINING RESULTS:")
    for set_name, set_metrics in metrics.items():
        logger.info(f"   {set_name.upper()}: R²={set_metrics['r2']:.4f}, "
                    f"RMSE={set_metrics['rmse']:.4f}, MAE={set_metrics['mae']:.4f}")

    # Enhanced overfitting analysis
    train_test_gap = metrics['train']['r2'] - metrics['test']['r2']
    train_val_gap = metrics['train']['r2'] - metrics['validation']['r2']
    test_val_gap = metrics['test']['r2'] - metrics['validation']['r2']

    logger.info(f"DATA ENHANCED OVERFITTING ANALYSIS:")
    logger.info(f"   Train-Test Gap: {train_test_gap:.4f}")
    logger.info(f"   Train-Validation Gap: {train_val_gap:.4f}")
    logger.info(f"   Test-Validation Gap: {test_val_gap:.4f}")

    if use_gap_optimization:
        logger.info(f"   CV Gap (from gap-optimized CV): {cv_result['r2_gap']:.4f}")

        # Check consistency between gaps
        if abs(cv_result['r2_gap'] - train_test_gap) > 0.05:
            logger.warning(f"WARNING CV gap and train-test gap differ significantly")
        else:
            logger.info(f"SUCCESS CV gap and train-test gap are consistent")

    # Overall overfitting assessment
    max_gap = max(train_test_gap, train_val_gap)
    if max_gap > 0.1:
        logger.warning("WARNING Potential overfitting detected (gap > 0.1)")
    else:
        logger.info("SUCCESS Good model generalization (gaps < 0.1)")

    # STEP 6: Additional cross-validation analysis (enhanced)
    logger.info("ROTATE STEP 6: Additional cross-validation analysis...")

    if use_gap_optimization:
        # We already have gap-optimized CV results
        logger.info(f"SUCCESS Gap-optimized CV completed:")
        logger.info(f"   CV R²: {cv_result['test_r2_mean']:.4f} +/- {cv_result['test_r2_std']:.4f}")
        logger.info(f"   Folds used: {cv_result['used_folds']}/{cv_result['total_folds']}")
    else:
        # Standard CV
        logger.info(f"SUCCESS Standard 5-Fold CV completed:")
        logger.info(f"   CV R²: {cv_result['test_r2_mean']:.4f} +/- {cv_result['test_r2_std']:.4f}")



    # STEP 7: Create enhanced visualizations
    logger.info("DATA STEP 7: Creating enhanced evaluation visualizations...")
    create_enhanced_retraining_visualizations(
        y_train_processed, y_train_pred,
        y_test_processed, y_test_pred,
        y_validation_processed, y_validation_pred,
        metrics, cv_result, retrain_dir, best_model_name, use_gap_optimization
    )

    # STEP 8: Feature importance analysis (same as before)
    logger.info("ANALYZING STEP 8: Feature importance analysis...")
    feature_importance = analyze_retrained_model_features(
        retrained_model, X_train_selected, selected_features, retrain_dir
    )

    # STEP 9: Save everything with enhanced results
    logger.info("SAVING STEP 9: Saving enhanced retrained model and results...")

    # Save the retrained model
    joblib.dump(retrained_model, os.path.join(retrain_dir, f'retrained_{best_model_name}.pkl'))

    # Save datasets
    X_train_selected.to_csv(os.path.join(retrain_dir, 'X_train_retrained.csv'), index=True)
    X_test_selected.to_csv(os.path.join(retrain_dir, 'X_test_retrained.csv'), index=True)
    X_validation_selected.to_csv(os.path.join(retrain_dir, 'X_validation_retrained.csv'), index=True)

    pd.Series(y_train_processed, name=target_column).to_csv(os.path.join(retrain_dir, 'y_train_retrained.csv'))
    pd.Series(y_test_processed, name=target_column).to_csv(os.path.join(retrain_dir, 'y_test_retrained.csv'))
    pd.Series(y_validation_processed, name=target_column).to_csv(
        os.path.join(retrain_dir, 'y_validation_retrained.csv'))

    # Run analysis
    results, analyzer = analyze_validation_performance(
        X_train=X_train_selected,  # Training features
        y_train=y_train_processed,  # Training targets
        X_val=X_validation_selected,  # Validation features
        y_val=y_validation_processed,  # Validation targets
        val_predictions=y_validation_pred,  # Your ensemble predictions
        save_dir='validation_analysis',
        target_threshold=0.0  # Focus on negative value bias
    )

    # Save comprehensive results
    retrain_results = {
        'model_name': best_model_name,
        'data_splits': {
            'train_indices': X_train_new.index.values.tolist(),
            'test_indices': X_test_new.index.values.tolist(),
            'validation_indices': validation_indices.tolist(),
            'split_sizes': {
                'train': len(X_train_new),
                'test': len(X_test_new),
                'validation': len(X_validation)
            }
        },
        'metrics': metrics,
        'cv_results': cv_result,
        'gap_optimization': {
            'enabled': use_gap_optimization,
            'cv_params': get_optimal_cv_params(best_model_name) if use_gap_optimization else None,
            'gap_threshold_met': cv_result['r2_gap'] <= get_optimal_cv_params(best_model_name)[
                'gap_threshold'] if use_gap_optimization and not np.isnan(cv_result['r2_gap']) else None
        },
        'overfitting_analysis': {
            'train_test_gap': train_test_gap,
            'train_validation_gap': train_val_gap,
            'test_validation_gap': test_val_gap,
            'max_gap': max_gap,
            'is_overfitting': max_gap > 0.1,
            'cv_gap': cv_result['r2_gap'] if not np.isnan(cv_result['r2_gap']) else None
        },
        'feature_importance': feature_importance,
        'selected_features': selected_features,
        'preprocessing_info': {
            'imputation_method': preprocessor.imputation_method if hasattr(preprocessor,
                                                                           'imputation_method') else 'unknown',
            'feature_engineering_enabled': best_results.get('feature_engineering_enabled', False)
        }
    }

    with open(os.path.join(retrain_dir, 'enhanced_retrain_results.json'), 'w') as f:
        def json_default(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        json.dump(retrain_results, f, indent=4, default=json_default)

    logger.info("=" * 60)
    logger.info("SUCCESS ENHANCED MODEL RETRAINING COMPLETED!")
    logger.info(f"FOLDER Results saved to: {retrain_dir}")
    logger.info(f"WINNER Best validation R²: {metrics['validation']['r2']:.4f}")
    if use_gap_optimization:
        logger.info(f"TARGET Gap-optimized CV R²: {cv_result['test_r2_mean']:.4f} +/- {cv_result['test_r2_std']:.4f}")
        logger.info(f"BALANCE CV Gap: {cv_result['r2_gap']:.4f}")
    logger.info("=" * 60)

    return {
        'retrained_model': retrained_model,
        'metrics': metrics,
        'cv_results': cv_result,
        'datasets': {
            'X_train': X_train_selected,
            'X_test': X_test_selected,
            'X_validation': X_validation_selected,
            'y_train': y_train_processed,
            'y_test': y_test_processed,
            'y_validation': y_validation_processed
        },
        'data_splits': retrain_results['data_splits'],
        'feature_importance': feature_importance,
        'retrain_dir': retrain_dir,
        'full_results': retrain_results,
        'gap_optimization_used': use_gap_optimization
    }

def retrain_with_explicit_control_enhanced(simple_results, iterative_results, experiment_dir,
                                          original_data, target_column='Bioaccessible Pb',
                                          imputation_method='iterative',
                                          model_name='lightgbm',
                                          validation_strategy='indices',
                                          validation_config=None,
                                          test_size=0.15,
                                          random_state=42,
                                          use_gap_optimization=True):  # New parameter

    logger.info("TARGET ENHANCED RETRAINING WITH GAP-OPTIMIZED CV")
    logger.info("=" * 50)
    logger.info(f"   Imputation: {imputation_method.upper()}")
    logger.info(f"   Model: {model_name.upper()}")
    logger.info(f"   Validation: {validation_strategy}")
    logger.info(f"   Gap Optimization: {'ENABLED' if use_gap_optimization else 'DISABLED'}")

    # Select results based on imputation method
    if imputation_method == 'simple':
        chosen_results = simple_results
    elif imputation_method == 'iterative':
        chosen_results = iterative_results
    else:
        raise ValueError("imputation_method must be 'simple' or 'iterative'")

    # Verify model exists
    if model_name not in chosen_results['models']:
        available_models = list(chosen_results['models'].keys())
        raise ValueError(f"Model '{model_name}' not found. Available: {available_models}")

    # Configure validation based on strategy
    validation_indices = None
    custom_validation_criteria = None
    validation_size = 0.15

    if validation_strategy == 'indices':
        if not isinstance(validation_config, (list, tuple, np.ndarray)):
            raise ValueError("For 'indices' strategy, validation_config must be a list of indices")
        validation_indices = list(validation_config)
        logger.info(f"   Using specific indices: {validation_indices}")

    elif validation_strategy == 'criteria':
        if not isinstance(validation_config, dict):
            raise ValueError("For 'criteria' strategy, validation_config must be a dict")
        custom_validation_criteria = validation_config
        logger.info(f"   Using criteria: {validation_config}")

    elif validation_strategy == 'random':
        if isinstance(validation_config, (int, float)):
            validation_size = validation_config
        logger.info(f"   Using random split: {validation_size * 100:.1f}%")

    else:
        raise ValueError("validation_strategy must be 'indices', 'criteria', or 'random'")

    # Call enhanced retraining function
    return retrain_best_model_with_gap_optimized_cv(
        best_results=chosen_results,
        best_model_name=model_name,
        experiment_dir=experiment_dir,
        original_data=original_data,
        target_column=target_column,
        validation_indices=validation_indices,
        custom_validation_criteria=custom_validation_criteria,
        validation_size=validation_size,
        test_size=test_size,
        random_state=random_state,
        use_gap_optimization=use_gap_optimization  # Pass the parameter
    )
def create_validation_mask(X, y, criteria):
    """
    Create a boolean mask for validation data based on custom criteria.

    Parameters:
    -----------
    X : DataFrame
        Features
    y : Series
        Target
    criteria : dict
        Selection criteria

    Returns:
    --------
    numpy.ndarray
        Boolean mask for validation samples
    """
    if 'column' not in criteria:
        raise ValueError("Criteria must include 'column' key")

    column = criteria['column']

    if column not in X.columns and column != y.name:
        raise ValueError(f"Column '{column}' not found in data")

    # Get the data to filter on
    if column in X.columns:
        data = X[column]
    else:
        data = y

    # Apply different types of criteria
    if 'values' in criteria:
        # Exact value matching
        mask = data.isin(criteria['values'])
    elif 'min_value' in criteria and 'max_value' in criteria:
        # Range filtering
        mask = (data >= criteria['min_value']) & (data <= criteria['max_value'])
    elif 'min_value' in criteria:
        # Minimum value
        mask = data >= criteria['min_value']
    elif 'max_value' in criteria:
        # Maximum value
        mask = data <= criteria['max_value']
    elif 'percentile_range' in criteria:
        # Percentile-based selection
        low, high = criteria['percentile_range']
        low_val = data.quantile(low / 100)
        high_val = data.quantile(high / 100)
        mask = (data >= low_val) & (data <= high_val)
    else:
        raise ValueError("No valid selection criteria provided")

    return mask.values


def create_enhanced_retraining_visualizations(y_train_true, y_train_pred, y_test_true, y_test_pred,
                                              y_val_true, y_val_pred, metrics, cv_result, save_dir,
                                              model_name, use_gap_optimization):
    """Create enhanced visualizations for retrained model evaluation."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: Prediction vs True (Train)
    ax1 = axes[0, 0]
    ax1.scatter(y_train_true, y_train_pred, alpha=0.6, s=30)
    min_val, max_val = min(y_train_true.min(), y_train_pred.min()), max(y_train_true.max(), y_train_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    ax1.set_xlabel('True Values',fontsize=18)
    ax1.set_ylabel('Predicted Values',fontsize=18)
    title = f'Training Set: R² = {metrics["train"]["r2"]:.3f}'
    if use_gap_optimization:
        title += ' (Gap-Optimized)'
    ax1.set_title(title,fontsize=20)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Prediction vs True (Test)
    ax2 = axes[0, 1]
    ax2.scatter(y_test_true, y_test_pred, alpha=0.6, s=30, color='orange')
    min_val, max_val = min(y_test_true.min(), y_test_pred.min()), max(y_test_true.max(), y_test_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    ax2.set_xlabel('True Values',fontsize=18)
    ax2.set_ylabel('Predicted Values',fontsize=18)
    ax2.set_title(f'Test Set: R² = {metrics["test"]["r2"]:.3f}',fontsize=20)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Prediction vs True (Validation)
    ax3 = axes[0, 2]
    ax3.scatter(y_val_true, y_val_pred, alpha=0.6, s=30, color='green')
    min_val, max_val = min(y_val_true.min(), y_val_pred.min()), max(y_val_true.max(), y_val_pred.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    ax3.set_xlabel('True Values',fontsize=18)
    ax3.set_ylabel('Predicted Values',fontsize=18)
    ax3.set_title(f'Validation Set: R² = {metrics["validation"]["r2"]:.3f}',fontsize=20)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Residuals (Train)
    ax4 = axes[1, 0]
    residuals_train = y_train_true - y_train_pred
    ax4.scatter(y_train_pred, residuals_train, alpha=0.6, s=30)
    ax4.axhline(y=0, color='r', linestyle='--', alpha=0.8)
    ax4.set_xlabel('Predicted Values',fontsize=18)
    ax4.set_ylabel('Residuals',fontsize=18)
    ax4.set_title(f'Training Residuals (RMSE = {metrics["train"]["rmse"]:.3f})',fontsize=22)
    ax4.grid(True, alpha=0.3)

    # Plot 5: Residuals (Test)
    ax5 = axes[1, 1]
    residuals_test = y_test_true - y_test_pred
    ax5.scatter(y_test_pred, residuals_test, alpha=0.6, s=30, color='orange')
    ax5.axhline(y=0, color='r', linestyle='--', alpha=0.8)
    ax5.set_xlabel('Predicted Values')
    ax5.set_ylabel('Residuals')
    ax5.set_title(f'Test Residuals (RMSE = {metrics["test"]["rmse"]:.3f})')
    ax5.grid(True, alpha=0.3)

    # Plot 6: Enhanced metrics comparison including CV results
    ax6 = axes[1, 2]
    sets = ['Train', 'Test', 'Validation']
    r2_scores = [metrics['train']['r2'], metrics['test']['r2'], metrics['validation']['r2']]
    colors = ['blue', 'orange', 'green']

    bars = ax6.bar(sets, r2_scores, color=colors, alpha=0.7)

    # Add CV result if available
    if use_gap_optimization and not np.isnan(cv_result['test_r2_mean']):
        ax6.axhline(y=cv_result['test_r2_mean'], color='red', linestyle='--',
                    label=f'CV R²: {cv_result["test_r2_mean"]:.3f}+/-{cv_result["test_r2_std"]:.3f}')
        ax6.legend()

    ax6.set_ylabel('R² Score')
    title = 'R² Comparison Across Sets'
    if use_gap_optimization:
        title += ' (Gap-Optimized)'
    ax6.set_title(title,fontsize=20)
    ax6.set_ylim(0, 1)
    ax6.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, score in zip(bars, r2_scores):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{score:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()

    filename = f'{model_name}_enhanced_retrain_evaluation'
    if use_gap_optimization:
        filename += '_gap_optimized'

    plt.savefig(os.path.join(save_dir, f'{filename}.png'), dpi=300, bbox_inches='tight')
    plt.close()


def analyze_retrained_model_features(model, X_train, feature_names, save_dir):
    """Analyze feature importance for the retrained model."""

    feature_importance = {}

    # Get feature importance if available
    if hasattr(model, 'feature_importances_'):
        importance_scores = model.feature_importances_
        feature_importance = dict(zip(feature_names, importance_scores))

        # Create feature importance plot
        plt.figure(figsize=(10, max(6, len(feature_names) * 0.4)))
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

        features, scores = zip(*sorted_features)
        y_pos = np.arange(len(features))

        plt.barh(y_pos, scores, color='skyblue', alpha=0.8)
        plt.yticks(y_pos, features)
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance for Retrained Model')
        plt.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, score in enumerate(scores):
            plt.text(score + max(scores) * 0.01, i, f'{score:.3f}',
                     va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'feature_importance_retrained.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"SUCCESS Feature importance analysis saved")
        logger.info(f"   Top 3 features: {list(features[:3])}")

    return feature_importance