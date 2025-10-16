# feature_importance.py - Module for generating feature importance visualizations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.inspection import permutation_importance
import logging

# Try to import shap for additional importance measures
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

logger = logging.getLogger(__name__)


class FeatureImportanceVisualizer:
    """Class for generating and visualizing feature importance metrics."""

    def __init__(self, experiment_dir):
        self.experiment_dir = experiment_dir
        self.viz_dir = os.path.join(experiment_dir, 'feature_importance')
        os.makedirs(self.viz_dir, exist_ok=True)

    def generate_importance_heatmap(self, models, X, y, top_n=10):
        """
        Generate a heatmap showing the top N most important features across multiple models.

        Parameters:
        -----------
        models : dict
            Dictionary of trained models with model names as keys
        X : DataFrame
            Feature matrix
        y : Series
            Target variable
        top_n : int
            Number of top features to include in the heatmap

        Returns:
        --------
        feature_importance_df : DataFrame
            DataFrame containing importance scores for each feature across models
        """
        logger.info(f"Generating feature importance heatmap for top {top_n} features")

        # Initialize dictionary to store feature importances for each model
        feature_importances = {}

        # Get feature importance from each model
        for model_name, model in models.items():
            # Skip models that don't support feature importance
            if model_name == 'neural_network':
                continue

            # Get feature importance based on model type
            if hasattr(model, 'feature_importances_'):
                # Tree-based models have feature_importances_ attribute
                importances = model.feature_importances_
                feature_importances[model_name] = dict(zip(X.columns, importances))

            elif SHAP_AVAILABLE:
                # Use SHAP for models that don't have direct feature importance
                try:
                    # Create explainer based on model type
                    if hasattr(model, 'predict_proba'):
                        explainer = shap.TreeExplainer(model)
                    else:
                        explainer = shap.Explainer(model)

                    # Calculate SHAP values
                    shap_values = explainer(X)

                    # Calculate mean absolute SHAP value for each feature
                    mean_shap = np.abs(shap_values.values).mean(axis=0)
                    feature_importances[model_name] = dict(zip(X.columns, mean_shap))

                except Exception as e:
                    logger.warning(f"Error calculating SHAP values for {model_name}: {str(e)}")
                    # Fallback to permutation importance
                    perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)
                    feature_importances[model_name] = dict(zip(X.columns, perm_importance.importances_mean))

            else:
                # Fallback to permutation importance for other models
                perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)
                feature_importances[model_name] = dict(zip(X.columns, perm_importance.importances_mean))

        # Convert to DataFrame
        feature_importance_df = pd.DataFrame(feature_importances)

        # Add average importance across models
        feature_importance_df['average'] = feature_importance_df.mean(axis=1)

        # Sort by average importance
        feature_importance_df = feature_importance_df.sort_values('average', ascending=False)

        # Select top N features
        top_features_df = feature_importance_df.head(top_n)

        # Normalize importance values for better visualization
        normalized_df = top_features_df.copy()
        for col in normalized_df.columns:
            normalized_df[col] = normalized_df[col] / normalized_df[col].max()

        # Generate heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(normalized_df.drop(columns=['average']), annot=True, cmap='viridis',
                    linewidths=0.5, fmt='.2f', cbar_kws={'label': 'Normalized Importance'})

        plt.title(f'Top {top_n} Most Important Features Across Models', fontsize=16)
        plt.ylabel('Features', fontsize=14)
        plt.xlabel('Models', fontsize=14)
        plt.tight_layout()

        # Save the heatmap
        heatmap_path = os.path.join(self.viz_dir, f'top_{top_n}_features_heatmap.png')
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Feature importance heatmap saved to {heatmap_path}")

        # Also create a bar chart for average importance
        plt.figure(figsize=(12, 8))
        top_features_df['average'].sort_values().plot(kind='barh', color='teal')

        plt.title(f'Average Importance of Top {top_n} Features', fontsize=16)
        plt.xlabel('Average Importance', fontsize=14)
        plt.ylabel('Features', fontsize=14)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Save the bar chart
        bar_path = os.path.join(self.viz_dir, f'top_{top_n}_features_bar.png')
        plt.savefig(bar_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Feature importance bar chart saved to {bar_path}")

        # Additional correlation heatmap between feature importance methods
        plt.figure(figsize=(10, 8))
        sns.heatmap(normalized_df.drop(columns=['average']).corr(),
                    annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)

        plt.title('Correlation Between Feature Importance Methods', fontsize=16)
        plt.tight_layout()

        # Save the correlation heatmap
        corr_path = os.path.join(self.viz_dir, 'feature_importance_correlation.png')
        plt.savefig(corr_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Save the feature importance data
        feature_importance_df.to_csv(os.path.join(self.viz_dir, 'feature_importance.csv'))

        return feature_importance_df

    def generate_pairwise_feature_interactions(self, model, X, y, top_n=10):
        """
        Generate heatmap showing pairwise feature interactions for top features.

        Parameters:
        -----------
        model : trained model object
            Tree-based model with feature_importances_
        X : DataFrame
            Feature matrix
        y : Series
            Target variable
        top_n : int
            Number of top features to analyze for interactions
        """
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model doesn't support built-in feature importance, using permutation importance")
            result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
            importances = result.importances_mean
        else:
            importances = model.feature_importances_

        # Get top N features
        feature_names = X.columns
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]

        # Create dataframe with only top features
        X_top = X[top_features]

        # Initialize interaction matrix
        interaction_matrix = np.zeros((top_n, top_n))

        # Calculate pairwise feature interaction strengths
        for i in range(top_n):
            for j in range(i + 1, top_n):
                feature_i = top_features[i]
                feature_j = top_features[j]

                # Create interaction term
                X_interaction = X_top.copy()
                X_interaction[f'{feature_i}_{feature_j}'] = X_interaction[feature_i] * X_interaction[feature_j]

                # Split data
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X_interaction, y, test_size=0.3, random_state=42)

                # Train model with and without interaction
                from sklearn.ensemble import RandomForestRegressor

                # Model without interaction
                model_without = RandomForestRegressor(n_estimators=100, random_state=42)
                model_without.fit(X_train.drop(f'{feature_i}_{feature_j}', axis=1), y_train)
                score_without = model_without.score(X_test.drop(f'{feature_i}_{feature_j}', axis=1), y_test)

                # Model with interaction
                model_with = RandomForestRegressor(n_estimators=100, random_state=42)
                model_with.fit(X_train, y_train)
                score_with = model_with.score(X_test, y_test)

                # Interaction strength is improvement in R²
                interaction_strength = score_with - score_without

                # Store in matrix
                interaction_matrix[i, j] = interaction_strength
                interaction_matrix[j, i] = interaction_strength

        # Create a mask for the upper triangle
        mask = np.triu(np.ones_like(interaction_matrix, dtype=bool), k=1)

        # Generate heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(interaction_matrix, mask=mask, cmap='viridis', annot=True, fmt='.4f',
                    xticklabels=top_features, yticklabels=top_features, linewidths=0.5)

        plt.title(f'Pairwise Feature Interactions Among Top {top_n} Features', fontsize=16)
        plt.tight_layout()

        # Save the heatmap
        interaction_path = os.path.join(self.viz_dir, 'feature_interactions_heatmap.png')
        plt.savefig(interaction_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Feature interaction heatmap saved to {interaction_path}")

        return interaction_matrix, top_features