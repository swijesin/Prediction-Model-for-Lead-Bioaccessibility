import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import logging

logger = logging.getLogger(__name__)


class WeightedMultiCriteriaModelSelector:
    """
    Enhanced multi-criteria model selector with custom weighting hierarchy:
    Prediction Accuracy > Model Performance (R²) > RMSE
    """

    def __init__(self, experiment_dir, custom_weights=None):
        """
        Initialize with custom weighting scheme.

        Parameters:
        -----------
        experiment_dir : str
            Directory to save results
        custom_weights : dict, optional
            Custom weights for criteria. Default prioritizes:
            - prediction_accuracy: 50% (highest priority)
            - model_performance: 35% (R² score)
            - rmse: 15% (lowest priority)
        """
        self.experiment_dir = experiment_dir
        self.results_dir = os.path.join(experiment_dir, 'weighted_multi_criteria_evaluation')
        os.makedirs(self.results_dir, exist_ok=True)
        self.evaluation_results = {}
        self.ranking_results = {}

        # Set custom weights with your hierarchy: Prediction Accuracy > R² > RMSE
        if custom_weights is None:
            self.weights = {
                'prediction_accuracy': 0.50,  # Highest priority - how well predictions correlate
                'model_performance': 0.35,  # Second priority - R² score
                'rmse': 0.15  # Lowest priority - prediction error magnitude
            }
        else:
            self.weights = custom_weights

        # Normalize weights to sum to 1
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}

        logger.info(f"Initialized WeightedMultiCriteriaModelSelector with weights:")
        logger.info(f"  Prediction Accuracy: {self.weights['prediction_accuracy']:.1%}")
        logger.info(f"  Model Performance (R²): {self.weights['model_performance']:.1%}")
        logger.info(f"  RMSE: {self.weights['rmse']:.1%}")

    def evaluate_models_comprehensive(self, models, X_train, X_test, y_train, y_test,
                                      use_cv=True, cross_validator=None):
        """
        Comprehensive evaluation of all models using weighted multi-criteria approach.
        """

        logger.info("Starting weighted multi-criteria model evaluation...")
        logger.info(f"Hierarchy: Prediction Accuracy ({self.weights['prediction_accuracy']:.1%}) > "
                    f"Model Performance ({self.weights['model_performance']:.1%}) > "
                    f"RMSE ({self.weights['rmse']:.1%})")

        comprehensive_results = {}

        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name}...")

            try:
                # Get predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                # Calculate comprehensive metrics
                metrics = self._calculate_comprehensive_metrics(
                    y_train, y_train_pred, y_test, y_test_pred
                )

                # Add cross-validation metrics if requested
                if use_cv and cross_validator is not None:
                    try:
                        cv_metrics = self._get_cv_metrics(
                            model, X_train, X_test, y_train, y_test, cross_validator, model_name
                        )
                        metrics.update(cv_metrics)
                    except Exception as e:
                        logger.warning(f"Could not get CV metrics for {model_name}: {str(e)}")
                        # Add default CV metrics
                        metrics.update({
                            'cv_train_r2_mean': np.nan,
                            'cv_train_r2_std': np.nan,
                            'cv_test_r2_mean': np.nan,
                            'cv_test_r2_std': np.nan,
                            'cv_r2_gap': np.nan,
                            'cv_used_folds': 0,
                            'cv_total_folds': 0
                        })

                # Store results
                comprehensive_results[model_name] = metrics

                # Generate detailed plots for this model
                self._generate_model_plots(y_test, y_test_pred, model_name)

                logger.info(f"COMPLETED {model_name}:")
                logger.info(f"  Test R²: {metrics['test_r2']:.4f}")
                logger.info(f"  RMSE: {metrics['test_rmse']:.4f}")
                logger.info(f"  Prediction Accuracy: {metrics['prediction_accuracy']:.4f}")

            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {str(e)}")
                # Create placeholder results
                comprehensive_results[model_name] = {
                    'test_r2': 0.0,
                    'test_rmse': 999.0,
                    'prediction_accuracy': 0.0,
                    'error': str(e)
                }

        self.evaluation_results = comprehensive_results

        # Create comprehensive comparison
        self._create_comprehensive_comparison()

        # Rank models using weighted multi-criteria approach
        try:
            ranking_results = self._rank_models_weighted_criteria(comprehensive_results)
            self.ranking_results = ranking_results

            # Generate final recommendations
            self._generate_final_recommendations(ranking_results)
        except Exception as e:
            logger.error(f"Error in ranking models: {str(e)}")
            self.ranking_results = {'summary': {'best_overall': {'model': list(models.keys())[0]}}}

        return comprehensive_results

    def _calculate_comprehensive_metrics(self, y_train, y_train_pred, y_test, y_test_pred):
        """Calculate comprehensive metrics for model evaluation."""

        try:
            # Standard regression metrics
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)

            # Prediction accuracy (correlation between actual and predicted)
            try:
                prediction_accuracy = np.corrcoef(y_test, y_test_pred)[0, 1]
                if np.isnan(prediction_accuracy):
                    prediction_accuracy = 0.0
            except:
                prediction_accuracy = 0.0

            # Additional useful metrics
            try:
                mape = self._calculate_mape(y_test, y_test_pred)
            except:
                mape = np.nan

            overfitting_gap = train_r2 - test_r2

            # Relative performance metrics
            try:
                relative_rmse = test_rmse / np.std(y_test) if np.std(y_test) > 0 else np.nan
            except:
                relative_rmse = np.nan

            # Residual analysis
            try:
                residuals = y_test - y_test_pred
                residual_std = np.std(residuals)
                residual_skewness = self._calculate_skewness(residuals)
            except:
                residual_std = np.nan
                residual_skewness = np.nan

            return {
                # Primary criteria (for weighted ranking)
                'test_r2': test_r2,
                'test_rmse': test_rmse,
                'prediction_accuracy': prediction_accuracy,

                # Secondary metrics
                'train_r2': train_r2,
                'train_rmse': train_rmse,
                'test_mae': test_mae,
                'train_mae': train_mae,
                'mape': mape,
                'overfitting_gap': overfitting_gap,
                'relative_rmse': relative_rmse,
                'residual_std': residual_std,
                'residual_skewness': residual_skewness,

                # Additional info
                'n_test_samples': len(y_test),
                'test_target_std': np.std(y_test),
                'test_target_mean': np.mean(y_test)
            }
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {
                'test_r2': 0.0,
                'test_rmse': 999.0,
                'prediction_accuracy': 0.0,
                'train_r2': 0.0,
                'train_rmse': 999.0,
                'test_mae': 999.0,
                'train_mae': 999.0,
                'mape': np.nan,
                'overfitting_gap': 999.0,
                'relative_rmse': np.nan,
                'residual_std': np.nan,
                'residual_skewness': np.nan,
                'n_test_samples': len(y_test),
                'test_target_std': np.std(y_test),
                'test_target_mean': np.mean(y_test)
            }

    def _rank_models_weighted_criteria(self, evaluation_results):
        """
        Rank models using weighted multi-criteria approach with your hierarchy.
        Prediction Accuracy (50%) > Model Performance/R² (35%) > RMSE (15%)
        """
        logger.info("Ranking models using WEIGHTED multi-criteria approach...")
        logger.info(f"Weight hierarchy: Prediction Accuracy > Model Performance > RMSE")

        # Filter out models with errors
        valid_results = {k: v for k, v in evaluation_results.items() if 'error' not in v}

        if not valid_results:
            logger.error("No valid model results for ranking")
            return {'summary': {'best_overall': {'model': 'None', 'weighted_score': 999}}}

        models = list(valid_results.keys())

        # Extract key metrics for ranking
        metrics_df = pd.DataFrame({
            'Model': models,
            'Test_R2': [valid_results[m]['test_r2'] for m in models],
            'Test_RMSE': [valid_results[m]['test_rmse'] for m in models],
            'Prediction_Accuracy': [valid_results[m]['prediction_accuracy'] for m in models],
            'Overfitting_Gap': [valid_results[m]['overfitting_gap'] for m in models],
            'MAPE': [valid_results[m]['mape'] for m in models]
        })

        # Individual rankings (1 = best, higher number = worse)
        metrics_df['R2_Rank'] = metrics_df['Test_R2'].rank(method='dense', ascending=False)
        metrics_df['RMSE_Rank'] = metrics_df['Test_RMSE'].rank(method='dense', ascending=True)  # Lower is better
        metrics_df['Accuracy_Rank'] = metrics_df['Prediction_Accuracy'].rank(method='dense', ascending=False)
        metrics_df['Gap_Rank'] = metrics_df['Overfitting_Gap'].rank(method='dense',
                                                                    ascending=True)  # Lower gap is better

        # WEIGHTED SCORING BASED ON YOUR HIERARCHY
        # Lower score = better (rank 1 gets lowest score)
        metrics_df['Weighted_Score'] = (
                self.weights['prediction_accuracy'] * metrics_df['Accuracy_Rank'] +
                self.weights['model_performance'] * metrics_df['R2_Rank'] +
                self.weights['rmse'] * metrics_df['RMSE_Rank']
        )

        # Alternative scoring strategies for comparison
        metrics_df['Equal_Weight_Score'] = (
                                                   metrics_df['R2_Rank'] +
                                                   metrics_df['RMSE_Rank'] +
                                                   metrics_df['Accuracy_Rank']
                                           ) / 3

        # Robust scoring (includes overfitting)
        metrics_df['Robust_Score'] = (
                self.weights['prediction_accuracy'] * metrics_df['Accuracy_Rank'] +
                self.weights['model_performance'] * metrics_df['R2_Rank'] +
                self.weights['rmse'] * metrics_df['RMSE_Rank'] +
                0.10 * metrics_df['Gap_Rank']  # Small penalty for overfitting
        )

        # Final rankings (lower score = better rank)
        metrics_df['Weighted_Rank'] = metrics_df['Weighted_Score'].rank(method='dense')
        metrics_df['Equal_Weight_Rank'] = metrics_df['Equal_Weight_Score'].rank(method='dense')
        metrics_df['Robust_Rank'] = metrics_df['Robust_Score'].rank(method='dense')

        # Sort by weighted score (primary recommendation based on your hierarchy)
        metrics_df = metrics_df.sort_values('Weighted_Score')

        # Save detailed ranking results
        try:
            metrics_df.to_csv(os.path.join(self.results_dir, 'weighted_model_rankings.csv'), index=False)
        except Exception as e:
            logger.error(f"Could not save rankings CSV: {str(e)}")

        # Create ranking summary
        best_model_name = metrics_df.iloc[0]['Model']
        ranking_summary = {
            'best_overall': {
                'model': best_model_name,
                'weighted_score': metrics_df.iloc[0]['Weighted_Score'],
                'test_r2': metrics_df.iloc[0]['Test_R2'],
                'test_rmse': metrics_df.iloc[0]['Test_RMSE'],
                'prediction_accuracy': metrics_df.iloc[0]['Prediction_Accuracy'],
                'rank_breakdown': {
                    'accuracy_rank': int(metrics_df.iloc[0]['Accuracy_Rank']),
                    'r2_rank': int(metrics_df.iloc[0]['R2_Rank']),
                    'rmse_rank': int(metrics_df.iloc[0]['RMSE_Rank'])
                }
            },
            'best_prediction_accuracy': {
                'model': metrics_df.loc[metrics_df['Accuracy_Rank'] == 1, 'Model'].iloc[0],
                'prediction_accuracy': metrics_df['Prediction_Accuracy'].max()
            },
            'best_r2': {
                'model': metrics_df.loc[metrics_df['R2_Rank'] == 1, 'Model'].iloc[0],
                'test_r2': metrics_df['Test_R2'].max()
            },
            'best_rmse': {
                'model': metrics_df.loc[metrics_df['RMSE_Rank'] == 1, 'Model'].iloc[0],
                'test_rmse': metrics_df['Test_RMSE'].min()
            }
        }

        # Log the results with weights
        logger.info("WEIGHTED MULTI-CRITERIA RANKING RESULTS:")
        logger.info(f"Weight Distribution: Pred.Acc.({self.weights['prediction_accuracy']:.1%}) > "
                    f"R²({self.weights['model_performance']:.1%}) > "
                    f"RMSE({self.weights['rmse']:.1%})")
        logger.info("")
        logger.info(
            f"WINNER: {ranking_summary['best_overall']['model']} (Weighted Score: {ranking_summary['best_overall']['weighted_score']:.2f})")

        breakdown = ranking_summary['best_overall']['rank_breakdown']
        logger.info(
            f"  Rank Breakdown: Pred.Acc.(#{breakdown['accuracy_rank']}) + R²(#{breakdown['r2_rank']}) + RMSE(#{breakdown['rmse_rank']})")
        logger.info(f"  Prediction Accuracy: {ranking_summary['best_overall']['prediction_accuracy']:.4f}")
        logger.info(f"  Model Performance (R²): {ranking_summary['best_overall']['test_r2']:.4f}")
        logger.info(f"  RMSE: {ranking_summary['best_overall']['test_rmse']:.4f}")
        logger.info("")

        logger.info("CATEGORY LEADERS:")
        logger.info(
            f"  Best Prediction Accuracy: {ranking_summary['best_prediction_accuracy']['model']} ({ranking_summary['best_prediction_accuracy']['prediction_accuracy']:.4f})")
        logger.info(f"  Best R²: {ranking_summary['best_r2']['model']} ({ranking_summary['best_r2']['test_r2']:.4f})")
        logger.info(
            f"  Best RMSE: {ranking_summary['best_rmse']['model']} ({ranking_summary['best_rmse']['test_rmse']:.4f})")

        return {
            'detailed_rankings': metrics_df,
            'summary': ranking_summary,
            'weights_used': self.weights
        }

    def _create_comprehensive_comparison(self):
        """Create comprehensive comparison visualizations with weight emphasis."""

        if not self.evaluation_results:
            logger.warning("No evaluation results available for comparison")
            return

        # Filter out models with errors
        valid_results = {k: v for k, v in self.evaluation_results.items() if 'error' not in v}

        if not valid_results:
            logger.warning("No valid results for comparison")
            return

        try:
            models = list(valid_results.keys())

            # Extract metrics
            test_r2 = [valid_results[m]['test_r2'] for m in models]
            test_rmse = [valid_results[m]['test_rmse'] for m in models]
            prediction_accuracy = [valid_results[m]['prediction_accuracy'] for m in models]

            # Create comparison plots with weight emphasis
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))

            # Plot 1: Prediction Accuracy (Highest Weight - 50%)
            ax1 = axes[0, 0]
            bars1 = ax1.bar(models, prediction_accuracy, color='gold', alpha=0.8)
            ax1.set_ylabel('Prediction Accuracy (Correlation)',fontsize=14)
            ax1.set_title(f'HIGHEST PRIORITY: Prediction Accuracy (Weight: {self.weights["prediction_accuracy"]:.1%})',fontsize=16)
            ax1.set_xticklabels(models, rotation=45)
            ax1.grid(True, alpha=0.3)

            # Add values on bars and highlight best
            best_acc_idx = np.argmax(prediction_accuracy)
            for i, (bar, value) in enumerate(zip(bars1, prediction_accuracy)):
                color = 'red' if i == best_acc_idx else 'black'
                weight = 'bold' if i == best_acc_idx else 'normal'
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{value:.3f}', ha='center', va='bottom', fontweight=weight, color=color)

            # Plot 2: R² Score (Second Priority - 35%)
            ax2 = axes[0, 1]
            bars2 = ax2.bar(models, test_r2, color='skyblue', alpha=0.8)
            ax2.set_ylabel('Test R² Score',fontsize=14)
            ax2.set_title(f'SECOND PRIORITY: Model Performance R² (Weight: {self.weights["model_performance"]:.1%})',fontsize=16)
            ax2.set_xticklabels(models, rotation=45)
            ax2.grid(True, alpha=0.3)

            # Add values on bars and highlight best
            best_r2_idx = np.argmax(test_r2)
            for i, (bar, value) in enumerate(zip(bars2, test_r2)):
                color = 'red' if i == best_r2_idx else 'black'
                weight = 'bold' if i == best_r2_idx else 'normal'
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{value:.3f}', ha='center', va='bottom', fontweight=weight, color=color)

            # Plot 3: RMSE (Lowest Priority - 15%)
            ax3 = axes[1, 0]
            bars3 = ax3.bar(models, test_rmse, color='lightcoral', alpha=0.8)
            ax3.set_ylabel('Test RMSE',fontsize=14)
            ax3.set_title(f'LOWEST PRIORITY: RMSE (Weight: {self.weights["rmse"]:.1%}) - Lower is Better',fontsize=16)
            ax3.set_xticklabels(models, rotation=45)
            ax3.grid(True, alpha=0.3)

            # Add values on bars and highlight best (lowest RMSE)
            best_rmse_idx = np.argmin(test_rmse)
            for i, (bar, value) in enumerate(zip(bars3, test_rmse)):
                color = 'red' if i == best_rmse_idx else 'black'
                weight = 'bold' if i == best_rmse_idx else 'normal'
                ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{value:.3f}', ha='center', va='bottom', fontweight=weight, color=color)

            # Plot 4: Weighted Score Visualization
            ax4 = axes[1, 1]

            # Calculate weighted scores for visualization
            if hasattr(self, 'ranking_results') and 'detailed_rankings' in self.ranking_results:
                rankings_df = self.ranking_results['detailed_rankings']
                weighted_scores = rankings_df['Weighted_Score'].values
                model_order = rankings_df['Model'].values

                # Reorder to match current model list
                score_dict = dict(zip(model_order, weighted_scores))
                ordered_scores = [score_dict.get(model, 999) for model in models]

                bars4 = ax4.bar(models, ordered_scores, color='mediumpurple', alpha=0.8)
                ax4.set_ylabel('Weighted Score (Lower = Better)',fontsize=14)
                ax4.set_title('Final Weighted Scores\n(Prediction Accuracy > R² > RMSE)',fontsize=16)
                ax4.set_xticklabels(models, rotation=45)
                ax4.grid(True, alpha=0.3)

                # Highlight winner (lowest score)
                best_weighted_idx = np.argmin(ordered_scores)
                for i, (bar, value) in enumerate(zip(bars4, ordered_scores)):
                    color = 'red' if i == best_weighted_idx else 'black'
                    weight = 'bold' if i == best_weighted_idx else 'normal'
                    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                             f'{value:.2f}', ha='center', va='bottom', fontweight=weight, color=color)
            else:
                ax4.text(0.5, 0.5, 'Weighted scores\nnot available',
                         transform=ax4.transAxes, ha='center', va='center')
                ax4.set_title('Weighted Scores (Not Available)')

            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'weighted_criteria_comparison.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.error(f"Error creating comprehensive comparison: {str(e)}")

    def _generate_final_recommendations(self, ranking_results):
        """Generate final recommendations with weight-aware analysis."""

        try:
            summary = ranking_results['summary']
            weights = ranking_results['weights_used']

            # Create recommendation text
            recommendations = []
            recommendations.append("WEIGHTED MULTI-CRITERIA MODEL SELECTION RESULTS")
            recommendations.append("=" * 60)
            recommendations.append("")
            recommendations.append(f"WEIGHTING HIERARCHY:")
            recommendations.append(
                f"  1. Prediction Accuracy: {weights['prediction_accuracy']:.1%} (Correlation between actual vs predicted)")
            recommendations.append(f"  2. Model Performance: {weights['model_performance']:.1%} (R² score)")
            recommendations.append(f"  3. RMSE: {weights['rmse']:.1%} (Prediction error magnitude)")
            recommendations.append("")

            # Overall best model
            best_overall = summary['best_overall']
            breakdown = best_overall['rank_breakdown']

            recommendations.append(f"RECOMMENDED MODEL: {best_overall['model']}")
            recommendations.append(f"   Weighted Score: {best_overall['weighted_score']:.2f} (lower = better)")
            recommendations.append(f"   Rank Breakdown:")
            recommendations.append(
                f"     Prediction Accuracy: #{breakdown['accuracy_rank']} ({best_overall['prediction_accuracy']:.4f})")
            recommendations.append(
                f"     Model Performance (R²): #{breakdown['r2_rank']} ({best_overall['test_r2']:.4f})")
            recommendations.append(f"     RMSE: #{breakdown['rmse_rank']} ({best_overall['test_rmse']:.4f})")
            recommendations.append("")

            # Category analysis
            recommendations.append("CATEGORY ANALYSIS:")

            # Check if winner dominates in highest priority category
            if best_overall['model'] == summary['best_prediction_accuracy']['model']:
                recommendations.append(
                    f"   EXCELLENT: Winner has BEST prediction accuracy ({weights['prediction_accuracy']:.1%} weight)")
            else:
                recommendations.append(
                    f"   NOTE: Best prediction accuracy is {summary['best_prediction_accuracy']['model']}")

            if best_overall['model'] == summary['best_r2']['model']:
                recommendations.append(f"   EXCELLENT: Winner also has BEST R² score")
            else:
                recommendations.append(f"   NOTE: Best R² score is {summary['best_r2']['model']}")

            if best_overall['model'] == summary['best_rmse']['model']:
                recommendations.append(f"   BONUS: Winner also has BEST RMSE")
            else:
                recommendations.append(f"   NOTE: Best RMSE is {summary['best_rmse']['model']}")

            recommendations.append("")

            # Strategy recommendations
            recommendations.append("STRATEGIC RECOMMENDATIONS:")

            # If winner is not best in highest priority category, explain trade-offs
            if best_overall['model'] != summary['best_prediction_accuracy']['model']:
                acc_leader = summary['best_prediction_accuracy']['model']
                recommendations.append(f"   TRADE-OFF ANALYSIS:")
                recommendations.append(
                    f"   - {best_overall['model']} (winner) offers balanced performance across all criteria")
                recommendations.append(
                    f"   - {acc_leader} has highest prediction accuracy but may be weaker in other areas")
                recommendations.append(
                    f"   - Given your {weights['prediction_accuracy']:.1%} weight on prediction accuracy, consider {acc_leader} if correlation is critical")
            else:
                recommendations.append(f"   OPTIMAL CHOICE: Winner excels in your highest priority criterion")

            recommendations.append("")
            recommendations.append("USAGE GUIDANCE:")
            recommendations.append(f"   - Use {best_overall['model']} for balanced, reliable predictions")
            recommendations.append(
                f"   - Prediction accuracy of {best_overall['prediction_accuracy']:.4f} indicates strong correlation")
            recommendations.append(f"   - R² of {best_overall['test_r2']:.4f} shows good explanatory power")
            recommendations.append(f"   - RMSE of {best_overall['test_rmse']:.4f} indicates prediction precision")

            recommendations.append("")
            recommendations.append("All detailed results saved to: " + self.results_dir)

            # Save recommendations to file
            try:
                with open(os.path.join(self.results_dir, 'weighted_model_recommendations.txt'), 'w',
                          encoding='utf-8') as f:
                    f.write('\n'.join(recommendations))
            except Exception as e:
                logger.error(f"Could not save recommendations file: {str(e)}")

            # Print to console
            for line in recommendations:
                logger.info(line)

        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")

    def set_custom_weights(self, prediction_accuracy=0.50, model_performance=0.35, rmse=0.15):
        """
        Update the weighting scheme.

        Parameters:
        -----------
        prediction_accuracy : float
            Weight for prediction accuracy (correlation)
        model_performance : float
            Weight for model performance (R²)
        rmse : float
            Weight for RMSE
        """
        self.weights = {
            'prediction_accuracy': prediction_accuracy,
            'model_performance': model_performance,
            'rmse': rmse
        }

        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}

        logger.info(f"Updated weights: Pred.Acc.({self.weights['prediction_accuracy']:.1%}) > "
                    f"R²({self.weights['model_performance']:.1%}) > "
                    f"RMSE({self.weights['rmse']:.1%})")

    # Include all the helper methods from the original class
    def _calculate_mape(self, y_true, y_pred, epsilon=1e-8):
        """Calculate Mean Absolute Percentage Error."""
        try:
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100
            return mape
        except:
            return np.nan

    def _calculate_skewness(self, data):
        """Calculate skewness of residuals."""
        try:
            data = np.array(data)
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0.0
            skewness = np.mean(((data - mean) / std) ** 3)
            return skewness
        except:
            return np.nan

    def _get_cv_metrics(self, model, X_train, X_test, y_train, y_test, cross_validator, model_name):
        """Get cross-validation metrics if available."""
        try:
            # Combine train and test for CV
            X_combined = pd.concat([X_train, X_test])
            y_combined = pd.concat([y_train, y_test])

            cv_result = cross_validator.evaluate_model(model, X_combined, y_combined, model_name)

            return {
                'cv_train_r2_mean': cv_result.get('train_r2_mean', np.nan),
                'cv_train_r2_std': cv_result.get('train_r2_std', np.nan),
                'cv_test_r2_mean': cv_result.get('test_r2_mean', np.nan),
                'cv_test_r2_std': cv_result.get('test_r2_std', np.nan),
                'cv_r2_gap': cv_result.get('r2_gap', np.nan),
                'cv_used_folds': cv_result.get('used_folds', 0),
                'cv_total_folds': cv_result.get('total_folds', 0)
            }
        except Exception as e:
            logger.warning(f"Could not get CV metrics for {model_name}: {str(e)}")
            return {
                'cv_train_r2_mean': np.nan,
                'cv_train_r2_std': np.nan,
                'cv_test_r2_mean': np.nan,
                'cv_test_r2_std': np.nan,
                'cv_r2_gap': np.nan,
                'cv_used_folds': 0,
                'cv_total_folds': 0
            }

    def _generate_model_plots(self, y_true, y_pred, model_name):
        """Generate detailed plots for each model."""

        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # Plot 1: Actual vs Predicted
            ax1 = axes[0, 0]
            ax1.scatter(y_true, y_pred, alpha=0.6, s=30)

            # Perfect prediction line
            min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
            ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)

            # Calculate and display metrics
            r2 = r2_score(y_true, y_pred)
            try:
                correlation = np.corrcoef(y_true, y_pred)[0, 1]
            except:
                correlation = 0.0

            ax1.set_xlabel('Actual Values',fontsize=14)
            ax1.set_ylabel('Predicted Values',fontsize=14)
            ax1.set_title(f'{model_name}: Actual vs Predicted\nR² = {r2:.4f}, Correlation = {correlation:.4f}',fontsize=16)
            ax1.grid(True, alpha=0.3)

            # Plot 2: Residuals vs Predicted
            ax2 = axes[0, 1]
            residuals = y_true - y_pred
            ax2.scatter(y_pred, residuals, alpha=0.6, s=30)
            ax2.axhline(y=0, color='r', linestyle='--', alpha=0.8)

            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            ax2.set_xlabel('Predicted Values',fontsize=14)
            ax2.set_ylabel('Residuals',fontsize=14)
            ax2.set_title(f'{model_name}: Residuals Plot\nRMSE = {rmse:.4f}',fontsize=16)
            ax2.grid(True, alpha=0.3)

            # Plot 3: Residuals Distribution
            ax3 = axes[1, 0]
            ax3.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
            ax3.axvline(x=0, color='r', linestyle='--', alpha=0.8)

            residual_mean = np.mean(residuals)
            residual_std = np.std(residuals)
            ax3.set_xlabel('Residuals',fontsize=14)
            ax3.set_ylabel('Frequency',fontsize=14)
            ax3.set_title(f'{model_name}: Residuals Distribution\nMean = {residual_mean:.4f}, Std = {residual_std:.4f}',fontsize=16)
            ax3.grid(True, alpha=0.3)

            # Plot 4: Weighted criteria visualization
            ax4 = axes[1, 1]

            # Show the three key metrics for this model
            metrics = ['Prediction\nAccuracy', 'R² Score', 'RMSE\n(inverted)']
            values = [correlation, r2, 1.0 - (rmse / (rmse + 1))]  # Invert RMSE for visualization
            weights = [self.weights['prediction_accuracy'], self.weights['model_performance'], self.weights['rmse']]

            # Create weighted bar chart
            colors = ['gold', 'skyblue', 'lightcoral']
            bars = ax4.bar(metrics, values, color=colors, alpha=0.8)

            # Add weight information
            for i, (bar, weight, value) in enumerate(zip(bars, weights, values)):
                ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                         f'{value:.3f}\n(Weight: {weight:.1%})',
                         ha='center', va='bottom', fontsize=9)

            ax4.set_ylabel('Normalized Score',fontsize=14)
            ax4.set_title(f'{model_name}: Weighted Criteria\n(Higher = Better)',fontsize=16)
            ax4.set_ylim(0, 1.2)
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f'{model_name}_weighted_analysis.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.error(f"Error creating plots for {model_name}: {str(e)}")

    def get_best_model_name(self):
        """Get the name of the best model based on weighted multi-criteria ranking."""
        try:
            if not self.ranking_results or 'summary' not in self.ranking_results:
                logger.warning("No ranking results available")
                return None

            return self.ranking_results['summary']['best_overall']['model']
        except Exception as e:
            logger.error(f"Error getting best model name: {str(e)}")
            return None

    def get_evaluation_summary(self):
        """Get a summary of all evaluation results with weighting information."""
        try:
            if not self.evaluation_results:
                return None

            # Filter out models with errors
            valid_results = {k: v for k, v in self.evaluation_results.items() if 'error' not in v}

            if not valid_results:
                return None

            summary = pd.DataFrame({
                model: {
                    'Prediction Accuracy': results['prediction_accuracy'],
                    'Test R²': results['test_r2'],
                    'Test RMSE': results['test_rmse'],
                    'Overfitting Gap': results['overfitting_gap']
                }
                for model, results in valid_results.items()
            }).T

            return summary
        except Exception as e:
            logger.error(f"Error creating evaluation summary: {str(e)}")
            return None

    def compare_weighting_strategies(self, evaluation_results):
        """
        Compare different weighting strategies to show impact of hierarchy.
        """
        logger.info("Comparing different weighting strategies...")

        # Define different weighting strategies
        strategies = {
            'Your_Hierarchy': {'prediction_accuracy': 0.50, 'model_performance': 0.35, 'rmse': 0.15},
            'Equal_Weights': {'prediction_accuracy': 0.33, 'model_performance': 0.33, 'rmse': 0.34},
            'R2_Priority': {'prediction_accuracy': 0.20, 'model_performance': 0.60, 'rmse': 0.20},
            'RMSE_Priority': {'prediction_accuracy': 0.20, 'model_performance': 0.20, 'rmse': 0.60}
        }

        results_comparison = {}

        for strategy_name, weights in strategies.items():
            # Temporarily change weights
            original_weights = self.weights.copy()
            self.weights = weights

            # Rank models with this strategy
            ranking = self._rank_models_weighted_criteria(evaluation_results)
            best_model = ranking['summary']['best_overall']['model']

            results_comparison[strategy_name] = {
                'best_model': best_model,
                'weights': weights.copy(),
                'weighted_score': ranking['summary']['best_overall']['weighted_score']
            }

            # Restore original weights
            self.weights = original_weights

        # Log comparison
        logger.info("WEIGHTING STRATEGY COMPARISON:")
        for strategy, result in results_comparison.items():
            w = result['weights']
            logger.info(f"  {strategy}: {result['best_model']} "
                        f"(Acc:{w['prediction_accuracy']:.1%}, R²:{w['model_performance']:.1%}, RMSE:{w['rmse']:.1%})")

        return results_comparison


# Integration function to replace the original multi-criteria selector in your main.py
def integrate_weighted_selector_in_main(models, X_train_selected, X_test_selected,
                                        y_train_processed, y_test_processed,
                                        imputer_dir, cross_validator=None):
    """
    Integration function to use the weighted selector in your main.py.

    Replace your existing multi-criteria selector call with this function.
    """

    logger.info("Using Weighted Multi-Criteria Model Selector")
    logger.info("Hierarchy: Prediction Accuracy > Model Performance > RMSE")

    # Initialize weighted selector with your hierarchy
    weighted_selector = WeightedMultiCriteriaModelSelector(
        experiment_dir=imputer_dir,
        custom_weights={
            'prediction_accuracy': 0.50,  # Highest priority - 50%
            'model_performance': 0.35,  # Second priority - 35%
            'rmse': 0.15  # Lowest priority - 15%
        }
    )

    # Evaluate models
    comprehensive_results = weighted_selector.evaluate_models_comprehensive(
        models=models,
        X_train=X_train_selected,
        X_test=X_test_selected,
        y_train=y_train_processed,
        y_test=y_test_processed,
        use_cv=True,
        cross_validator=cross_validator
    )

    # Get best model
    best_model_name = weighted_selector.get_best_model_name()

    # Optional: Compare different weighting strategies
    strategy_comparison = weighted_selector.compare_weighting_strategies(comprehensive_results)

    logger.info(f"WINNER: {best_model_name} selected using weighted hierarchy")

    return {
        'weighted_selector': weighted_selector,
        'comprehensive_results': comprehensive_results,
        'best_model_name': best_model_name,
        'strategy_comparison': strategy_comparison
    }


def compare_weighted_results(simple_results, iterative_results, experiment_dir):
    """
    Compare weighted multi-criteria results from different imputation methods.

    Parameters:
    -----------
    simple_results : dict
        Results from simple imputation pipeline
    iterative_results : dict
        Results from iterative imputation pipeline
    experiment_dir : str
        Base experiment directory

    Returns:
    --------
    dict
        Comparison results with winner information
    """
    logger.info("Comparing weighted multi-criteria results between imputation methods...")

    # Create comparison directory
    comparison_dir = os.path.join(experiment_dir, 'weighted_comparison')
    os.makedirs(comparison_dir, exist_ok=True)

    try:
        # Extract weighted selector objects and results
        simple_selector = simple_results.get('weighted_selector')
        iterative_selector = iterative_results.get('weighted_selector')

        simple_best = simple_results.get('best_model_name')
        iterative_best = iterative_results.get('best_model_name')

        # Get evaluation results
        if simple_selector and hasattr(simple_selector, 'evaluation_results'):
            simple_eval_results = simple_selector.evaluation_results
        else:
            simple_eval_results = simple_results.get('evaluation_results', {})

        if iterative_selector and hasattr(iterative_selector, 'evaluation_results'):
            iterative_eval_results = iterative_selector.evaluation_results
        else:
            iterative_eval_results = iterative_results.get('evaluation_results', {})

        logger.info("IMPUTATION METHOD COMPARISON:")
        logger.info(f"   Simple Imputation Best: {simple_best}")
        logger.info(f"   Iterative Imputation Best: {iterative_best}")

        # Extract metrics for best models
        simple_metrics = simple_eval_results.get(simple_best, {})
        iterative_metrics = iterative_eval_results.get(iterative_best, {})

        # Create comparison using the same weights as WeightedMultiCriteriaModelSelector
        weights = {
            'prediction_accuracy': 0.50,
            'model_performance': 0.35,
            'rmse': 0.15
        }

        # Calculate weighted scores
        simple_score = _calculate_comparison_score(simple_metrics, weights)
        iterative_score = _calculate_comparison_score(iterative_metrics, weights)

        # Determine winner
        if simple_score > iterative_score:
            winner = 'Simple'
            winner_model = simple_best
            winner_results = simple_results
            winner_score = simple_score
        else:
            winner = 'Iterative'
            winner_model = iterative_best
            winner_results = iterative_results
            winner_score = iterative_score

        logger.info(f"WINNER: {winner} Imputation with {winner_model} model")
        logger.info(f"   Weighted Score: {winner_score:.4f}")

        # Create comparison visualizations
        _create_imputation_comparison_charts(
            simple_metrics, iterative_metrics,
            simple_best, iterative_best,
            weights, comparison_dir
        )

        # Save comparison data
        comparison_data = {
            'Imputation Method': ['Simple', 'Iterative'],
            'Best Model': [simple_best, iterative_best],
            'Test R²': [simple_metrics.get('test_r2', 0.0), iterative_metrics.get('test_r2', 0.0)],
            'Test RMSE': [simple_metrics.get('test_rmse', 999.0), iterative_metrics.get('test_rmse', 999.0)],
            'Prediction Accuracy': [simple_metrics.get('prediction_accuracy', 0.0),
                                    iterative_metrics.get('prediction_accuracy', 0.0)],
            'Weighted Score': [simple_score, iterative_score]
        }

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(os.path.join(comparison_dir, 'imputation_comparison.csv'), index=False)

        return {
            'winner_method': winner,
            'winner_model': winner_model,
            'winner_results': winner_results,
            'comparison_df': comparison_df,
            'comparison_dir': comparison_dir,
            'simple_score': simple_score,
            'iterative_score': iterative_score,
            'weights_used': weights
        }

    except Exception as e:
        logger.error(f"Error in compare_weighted_results: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")

        # Return safe fallback
        return {
            'winner_method': 'Simple',
            'winner_model': simple_results.get('best_model_name', 'random_forest'),
            'winner_results': simple_results,
            'comparison_df': pd.DataFrame(),
            'comparison_dir': comparison_dir,
            'error': str(e)
        }


def _calculate_comparison_score(metrics, weights):
    """
    Calculate weighted comparison score for a model's metrics.
    Higher score is better.
    """
    try:
        # Extract metrics with safe defaults
        pred_acc = metrics.get('prediction_accuracy', 0.0)
        r2 = max(0.0, metrics.get('test_r2', 0.0))  # Floor at 0
        rmse = metrics.get('test_rmse', 999.0)

        # Normalize RMSE to 0-1 scale (lower RMSE = higher score)
        rmse_score = 1.0 / (1.0 + rmse)

        # Calculate weighted score
        weighted_score = (
                weights['prediction_accuracy'] * pred_acc +
                weights['model_performance'] * r2 +
                weights['rmse'] * rmse_score
        )

        return weighted_score

    except Exception:
        return 0.0


def _create_imputation_comparison_charts(simple_metrics, iterative_metrics,
                                         simple_best, iterative_best,
                                         weights, comparison_dir):
    """
    Create detailed comparison charts for imputation methods.
    """
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Individual metrics comparison
        ax1 = axes[0, 0]
        metrics_names = ['Prediction\nAccuracy', 'Test R²', 'Test RMSE']

        simple_values = [
            simple_metrics.get('prediction_accuracy', 0.0),
            simple_metrics.get('test_r2', 0.0),
            simple_metrics.get('test_rmse', 999.0)
        ]

        iterative_values = [
            iterative_metrics.get('prediction_accuracy', 0.0),
            iterative_metrics.get('test_r2', 0.0),
            iterative_metrics.get('test_rmse', 999.0)
        ]

        x = np.arange(len(metrics_names))
        width = 0.35

        bars1 = ax1.bar(x - width / 2, simple_values, width, label='Simple', alpha=0.7, color='lightblue')
        bars2 = ax1.bar(x + width / 2, iterative_values, width, label='Iterative', alpha=0.7, color='lightcoral')

        ax1.set_ylabel('Metric Value',fontsize=18)
        ax1.set_title('Individual Metrics Comparison',fontsize=20)
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics_names,fontsize=16)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add value labels
        for bars, values in [(bars1, simple_values), (bars2, iterative_values)]:
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2., height + abs(height) * 0.02,
                         f'{value:.3f}', ha='center', va='bottom', fontsize=9)

        # Plot 2: Weighted importance
        ax2 = axes[0, 1]
        weight_names = ['Prediction\nAccuracy\n(50%)', 'Model Performance\n(R²)\n(35%)', 'RMSE\n(15%)']
        weight_values = [weights['prediction_accuracy'], weights['model_performance'], weights['rmse']]
        colors = ['gold', 'skyblue', 'lightcoral']

        wedges, texts, autotexts = ax2.pie(weight_values, labels=weight_names, colors=colors,
                                           autopct='%1.1f%%', startangle=90)
        ax2.set_title('Weighting Hierarchy\n(Prediction Accuracy > R² > RMSE)',fontsize=20)

        # Plot 3: Final weighted scores
        ax3 = axes[1, 0]

        simple_score = _calculate_comparison_score(simple_metrics, weights)
        iterative_score = _calculate_comparison_score(iterative_metrics, weights)

        methods = ['Simple', 'Iterative']
        scores = [simple_score, iterative_score]
        colors = ['lightblue', 'lightcoral']

        bars = ax3.bar(methods, scores, color=colors, alpha=0.8)
        ax3.set_ylabel('Weighted Score (Higher = Better)',fontsize=18)
        ax3.set_title('Final Weighted Comparison',fontsize=20)
        ax3.grid(True, alpha=0.3)

        # Highlight winner
        winner_idx = 0 if simple_score > iterative_score else 1
        bars[winner_idx].set_color('gold')
        bars[winner_idx].set_alpha(1.0)
        bars[winner_idx].set_edgecolor('darkred')
        bars[winner_idx].set_linewidth(3)

        # Add value labels
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{score:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=12)

        # Plot 4: Summary table
        ax4 = axes[1, 1]
        ax4.axis('off')

        # Create summary table
        summary_data = [
            ['Metric', 'Simple', 'Iterative', 'Winner'],
            ['Prediction Accuracy', f"{simple_metrics.get('prediction_accuracy', 0.0):.4f}",
             f"{iterative_metrics.get('prediction_accuracy', 0.0):.4f}",
             'Simple' if simple_metrics.get('prediction_accuracy', 0.0) > iterative_metrics.get('prediction_accuracy',
                                                                                                0.0) else 'Iterative'],
            ['Test R²', f"{simple_metrics.get('test_r2', 0.0):.4f}",
             f"{iterative_metrics.get('test_r2', 0.0):.4f}",
             'Simple' if simple_metrics.get('test_r2', 0.0) > iterative_metrics.get('test_r2', 0.0) else 'Iterative'],
            ['Test RMSE', f"{simple_metrics.get('test_rmse', 999.0):.4f}",
             f"{iterative_metrics.get('test_rmse', 999.0):.4f}",
             'Simple' if simple_metrics.get('test_rmse', 999.0) < iterative_metrics.get('test_rmse',
                                                                                        999.0) else 'Iterative'],
            ['Weighted Score', f"{simple_score:.4f}", f"{iterative_score:.4f}",
             'Simple' if simple_score > iterative_score else 'Iterative'],
            ['Best Model', simple_best, iterative_best, '']
        ]

        table = ax4.table(cellText=summary_data[1:], colLabels=summary_data[0],
                          cellLoc='center', loc='center', bbox=[0, 0, 1, 1])

        table.auto_set_font_size(False)
        table.set_fontsize(18)
        table.scale(1.2, 2)

        # Style the table
        for i in range(len(summary_data)):
            for j in range(len(summary_data[0])):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#40466e')
                    cell.set_text_props(weight='bold', color='white')
                elif j == 3 and i < len(summary_data) - 1:  # Winner column
                    if summary_data[i][3] in ['Simple', 'Iterative']:
                        cell.set_facecolor('#90EE90' if summary_data[i][3] != '' else '#f8f9fa')
                        cell.set_text_props(weight='bold')
                else:
                    cell.set_facecolor('#f8f9fa')

        ax4.set_title('Detailed Comparison Summary', fontweight='bold', pad=20,fontsize=22)

        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, 'imputation_method_comparison.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Comparison charts saved to {comparison_dir}")

    except Exception as e:
        logger.error(f"Error creating comparison charts: {str(e)}")


# Update the integrate_weighted_selector_in_main function to store results properly
def integrate_weighted_selector_in_main(models, X_train_selected, X_test_selected,
                                        y_train_processed, y_test_processed,
                                        imputer_dir, cross_validator=None):
    """
    Integration function to use the weighted selector in your main.py.
    Updated to return results in the expected format.
    """
    logger.info("Using Weighted Multi-Criteria Model Selector")
    logger.info("Hierarchy: Prediction Accuracy > Model Performance > RMSE")

    # Initialize weighted selector
    weighted_selector = WeightedMultiCriteriaModelSelector(
        experiment_dir=imputer_dir,
        custom_weights={
            'prediction_accuracy': 0.50,
            'model_performance': 0.35,
            'rmse': 0.15
        }
    )

    # Evaluate models
    comprehensive_results = weighted_selector.evaluate_models_comprehensive(
        models=models,
        X_train=X_train_selected,
        X_test=X_test_selected,
        y_train=y_train_processed,
        y_test=y_test_processed,
        use_cv=True,
        cross_validator=cross_validator
    )

    # Get best model
    best_model_name = weighted_selector.get_best_model_name()

    logger.info(f"WINNER: {best_model_name} selected using weighted hierarchy")

    # Return results in format expected by main.py
    return {
        'weighted_selector': weighted_selector,
        'evaluation_results': comprehensive_results,
        'best_model_name': best_model_name,
        'models': models,  # Include original models
        # Add these for backward compatibility
        best_model_name: comprehensive_results.get(best_model_name, {}),
        # Add all individual model results for compatibility
        **comprehensive_results
    }

def compare_multi_criteria_results(simple_results, iterative_results, experiment_dir):
    """
    Simple wrapper function for main.py - just calls compare_weighted_results.
    This maintains the same function name your main.py expects.
    """
    return compare_weighted_results(simple_results, iterative_results, experiment_dir)