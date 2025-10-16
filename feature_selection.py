import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
import lightgbm as lgb
from cross_validation import create_stratified_kfold_for_regression
import matplotlib.pyplot as plt
import os
import joblib
import logging

logger = logging.getLogger(__name__)


class FeatureSelector:
    """Robust class for performing feature selection with NaN handling."""

    def __init__(self, methods=None, random_state=42, experiment_dir=None, k_best=12):
        self.methods = methods if methods is not None else ['variance', 'mutual_info', 'random_forest']
        self.random_state = random_state
        self.experiment_dir = experiment_dir
        self.selectors = {}
        self.selected_features_per_method = {}
        self.feature_votes = None
        self.selected_features = None
        self.k_best = k_best

    def fit(self, X, y):
        """Fit multiple feature selection methods and combine results with robust NaN handling."""
        # First, check for and handle any remaining NaN values
        self._check_and_handle_nans(X, y)

        feature_names = X.columns.tolist()
        n_features = len(feature_names)

        # Adjust k_best based on the number of features
        self.k_best = min(self.k_best, n_features - 1)

        # Initialize vote counter
        feature_votes = pd.DataFrame(0, index=feature_names, columns=['votes'])

        # 1. Variance Threshold with NaN handling
        if 'variance' in self.methods:
            logger.info("Running variance threshold feature selection...")
            try:
                # Replace NaN with column mean temporarily for variance calculation
                X_no_nan = X.fillna(X.mean())

                selector = VarianceThreshold(threshold=0.01)
                selector.fit(X_no_nan)

                # Get selected features
                support = selector.get_support()
                selected_features = [feature for feature, selected in zip(feature_names, support) if selected]
                self.selected_features_per_method['variance'] = selected_features

                # Add votes
                feature_votes.loc[selected_features, 'votes'] += 1

                # Store selector
                self.selectors['variance'] = selector

            except Exception as e:
                logger.error(f"Error in variance threshold selection: {str(e)}")
                logger.info("Skipping variance threshold method due to error")

        # 2. Mutual Information with NaN handling
        if 'mutual_info' in self.methods:
            logger.info("Running mutual information feature selection...")
            try:
                # Handle potential NaN values
                X_no_nan = X.fillna(X.mean())
                y_no_nan = y.fillna(y.mean()) if isinstance(y, pd.Series) else y

                selector = SelectKBest(mutual_info_regression, k=self.k_best)
                selector.fit(X_no_nan, y_no_nan)

                # Get selected features
                support = selector.get_support()
                selected_features = [feature for feature, selected in zip(feature_names, support) if selected]
                self.selected_features_per_method['mutual_info'] = selected_features

                # Add votes
                feature_votes.loc[selected_features, 'votes'] += 1

                # Store selector
                self.selectors['mutual_info'] = selector

                # Plot feature importances
                if self.experiment_dir:
                    scores = selector.scores_
                    plt.figure(figsize=(10, 6))
                    plt.bar(range(len(scores)), scores)
                    plt.xticks(range(len(scores)), feature_names, rotation=90)
                    plt.title('Mutual Information Feature Importance')
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.experiment_dir, 'mutual_info_importance.png'))
                    plt.close()

            except Exception as e:
                logger.error(f"Error in mutual information selection: {str(e)}")
                logger.info("Skipping mutual information method due to error")

        # 3. Random Forest feature importance with NaN handling
        if 'random_forest' in self.methods:
            logger.info("Running Random Forest feature importance...")
            try:
                # Handle potential NaN values
                X_no_nan = X.fillna(X.mean())
                y_no_nan = y.fillna(y.mean()) if isinstance(y, pd.Series) else y

                model = RandomForestRegressor(
                    n_estimators=100,
                    random_state=self.random_state,
                    max_depth=10,  # Limit depth for speed and to prevent overfitting
                    n_jobs=-1  # Use all cores
                )
                model.fit(X_no_nan, y_no_nan)

                # Get feature importances
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]

                # Select top k features
                top_k_indices = indices[:self.k_best]
                support = np.zeros(len(feature_names), dtype=bool)
                support[top_k_indices] = True

                selected_features = [feature for feature, selected in zip(feature_names, support) if selected]
                self.selected_features_per_method['random_forest'] = selected_features

                # Add votes
                feature_votes.loc[selected_features, 'votes'] += 1

                # Store model as selector
                self.selectors['random_forest'] = model

                # Plot feature importances
                if self.experiment_dir:
                    plt.figure(figsize=(10, 6))
                    plt.bar(range(len(importances)), importances[indices])
                    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
                    plt.title('Random Forest Feature Importance')
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.experiment_dir, 'random_forest_importance.png'))
                    plt.close()

            except Exception as e:
                logger.error(f"Error in random forest importance: {str(e)}")
                logger.info("Skipping random forest method due to error")

        # Store feature votes
        self.feature_votes = feature_votes

        # Select features that got votes from at least half of the methods
        methods_used = sum(1 for method in self.methods if method in self.selected_features_per_method)
        min_votes = max(1, methods_used // 2)
        self.selected_features = feature_votes[feature_votes['votes'] >= min_votes].index.tolist()

        # If too few features selected, take the top K based on votes
        if len(self.selected_features) < self.k_best:
            self.selected_features = feature_votes.sort_values('votes', ascending=False).head(
                self.k_best).index.tolist()

        # If still no features selected (possible if all methods failed), use all features
        if not self.selected_features:
            logger.warning("No features selected by any method. Using all features.")
            self.selected_features = feature_names

        logger.info(f"Selected {len(self.selected_features)} features: {self.selected_features}")

        # Save results
        if self.experiment_dir:
            # Save feature votes
            feature_votes.to_csv(os.path.join(self.experiment_dir, 'feature_votes.csv'))

            # Save selected features
            with open(os.path.join(self.experiment_dir, 'selected_features.txt'), 'w') as f:
                for feature in self.selected_features:
                    f.write(f"{feature}\n")

            # Save feature selector
            joblib.dump(self, os.path.join(self.experiment_dir, 'feature_selector.pkl'))

        return self

    def _check_and_handle_nans(self, X, y):
        """Check for and handle NaN values in the input data."""
        X_null_count = X.isnull().sum().sum()
        y_null_count = y.isnull().sum() if hasattr(y, 'isnull') else 0

        if X_null_count > 0:
            logger.warning(f"Found {X_null_count} NaN values in X. These will be handled internally.")

        if y_null_count > 0:
            logger.warning(f"Found {y_null_count} NaN values in y. These will be handled internally.")

    def transform(self, X):
        """Apply feature selection to input data."""
        if self.selected_features is None:
            raise ValueError("Feature selector has not been fit yet")

        # Select only the chosen features
        X_selected = X[self.selected_features]

        # Final check - if any selected columns still have NaNs, fill with mean
        if X_selected.isnull().sum().sum() > 0:
            logger.warning("NaN values detected in selected features. Filling with column means.")
            X_selected = X_selected.fillna(X_selected.mean())

        return X_selected

    def fit_transform(self, X, y):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def get_selected_features(self):
        """Return the list of selected features."""
        return self.selected_features

    def select_features_with_limit(self, X, y, max_features=10):
        """
        Select features with a hard limit on the maximum number of features.

        Parameters:
        -----------
        X : DataFrame
            Feature matrix
        y : Series
            Target values
        max_features : int, default=10
            Maximum number of features to select

        Returns:
        --------
        selected_features : list
            Names of selected features
        """
        from sklearn.feature_selection import mutual_info_regression, f_regression
        import numpy as np

        logger.info(f"Selecting a maximum of {max_features} features...")

        # Calculate feature importance scores using multiple methods
        # Method 1: Mutual Information
        mi_scores = mutual_info_regression(X, y, random_state=self.random_state)
        mi_features = [(X.columns[i], mi_scores[i]) for i in range(len(X.columns))]
        mi_features.sort(key=lambda x: x[1], reverse=True)

        # Method 2: F-statistic
        f_scores, _ = f_regression(X, y)
        f_features = [(X.columns[i], f_scores[i]) for i in range(len(X.columns))]
        f_features.sort(key=lambda x: x[1], reverse=True)

        # Method 3: Random Forest importance (if available)
        try:
            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
            rf.fit(X, y)
            rf_features = [(X.columns[i], rf.feature_importances_[i]) for i in range(len(X.columns))]
            rf_features.sort(key=lambda x: x[1], reverse=True)
        except:
            rf_features = []

        # Combine methods with weighted voting
        feature_votes = {}

        # Add each feature with its normalized rank
        for i, (feature, _) in enumerate(mi_features):
            feature_votes[feature] = feature_votes.get(feature, 0) + (len(mi_features) - i) / len(mi_features)

        for i, (feature, _) in enumerate(f_features):
            feature_votes[feature] = feature_votes.get(feature, 0) + (len(f_features) - i) / len(f_features)

        if rf_features:
            for i, (feature, _) in enumerate(rf_features):
                feature_votes[feature] = feature_votes.get(feature, 0) + (len(rf_features) - i) / len(rf_features)

        # Sort features by combined votes
        sorted_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)

        # Select top features up to max_features
        selected_features = [feature for feature, _ in sorted_features[:max_features]]

        logger.info(f"Selected {len(selected_features)} features: {selected_features}")

        return selected_features

    def recursive_feature_selection_with_cv(self, X, y, estimator=None, cv=5, scoring='r2'):
        """
        Perform recursive feature elimination with cross-validation to find the optimal number of features.

        Parameters:
        -----------
        X : DataFrame
            Feature matrix
        y : Series
            Target values
        estimator : estimator object, default=None
            Base estimator used for feature ranking. If None, uses a RandomForestRegressor.
        cv : int, cross-validation generator or iterable, default=5
            CV strategy for feature selection
        scoring : str, callable, default='r2'
            Scoring metric to use for feature selection

        Returns:
        --------
        selected_features : list
            List of selected feature names
        """

        logger.info("Performing recursive feature elimination with cross-validation...")

        # Create estimator if not provided
        if estimator is None:
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)

        # Create stratified CV folds for regression (using your existing function)
        cv_obj, y_binned = create_stratified_kfold_for_regression(y, n_splits=cv, random_state=42)

        # Create RFECV object
        rfecv = RFECV(
            estimator=estimator,
            step=1,
            cv=cv_obj.split(X, y_binned),
            scoring=scoring,
            min_features_to_select=3,
            n_jobs=-1,
            verbose=1
        )

        # Fit RFECV
        rfecv.fit(X, y)

        # Get selected features
        selected_features = X.columns[rfecv.support_].tolist()

        # Log results
        logger.info(f"RFECV selected {len(selected_features)} features: {selected_features}")
        logger.info(f"Optimal number of features: {rfecv.n_features_}")
        logger.info(f"Feature ranking: {rfecv.ranking_}")

        # Plot number of features vs. CV score
        plt.figure(figsize=(10, 6))
        plt.xlabel("Number of features selected")
        plt.ylabel(f"Cross-validation {scoring}")
        plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1),
                 rfecv.cv_results_['mean_test_score'])

        # Add confidence interval
        plt.fill_between(range(1, len(rfecv.cv_results_['mean_test_score']) + 1),
                         rfecv.cv_results_['mean_test_score'] - rfecv.cv_results_['std_test_score'],
                         rfecv.cv_results_['mean_test_score'] + rfecv.cv_results_['std_test_score'],
                         alpha=0.3)

        # Add vertical line at optimal number of features
        plt.axvline(x=rfecv.n_features_, color='r', linestyle='--',
                    label=f'Optimal number of features: {rfecv.n_features_}')

        plt.legend()
        plt.title("Recursive Feature Elimination with Cross-Validation")
        plt.tight_layout()

        # Save the plot
        plt.savefig(os.path.join(self.experiment_dir, 'rfecv_plot.png'))
        plt.close()

        # Feature importance ranking
        feat_importances = []
        for i, col in enumerate(X.columns):
            feat_importances.append((col, rfecv.ranking_[i]))

        # Sort by ranking (lower is better)
        feat_importances.sort(key=lambda x: x[1])

        # Plot feature importance
        plt.figure(figsize=(12, 8))
        features = [f[0] for f in feat_importances]
        rankings = [f[1] for f in feat_importances]

        plt.barh(range(len(features)), rankings, align='center')
        plt.yticks(range(len(features)), features)
        plt.xlabel('Ranking (1 = Best)')
        plt.title('Feature Ranking by RFECV')
        plt.tight_layout()
        plt.savefig(os.path.join(self.experiment_dir, 'rfecv_feature_ranking.png'))
        plt.close()

        return selected_features

    def multi_model_feature_selection(self, X, y, cv=5):
        """
        Perform feature selection using multiple models and find consensus features.

        Parameters:
        -----------
        X : DataFrame
            Feature matrix
        y : Series
            Target values
        cv : int, default=5
            Number of CV folds

        Returns:
        --------
        consensus_features : list
            List of features selected by majority of models
        """



        logger.info("Performing multi-model feature selection...")

        # Define models for feature selection
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'LassoCV': LassoCV(cv=cv, random_state=42),
            'LightGBM': lgb.LGBMRegressor(random_state=42)
        }

        # Create CV folds
        cv_obj, y_binned = create_stratified_kfold_for_regression(y, n_splits=cv, random_state=42)
        cv_indices = list(cv_obj.split(X, y_binned))

        # Track selected features for each model
        selected_features = {}
        feature_votes = {col: 0 for col in X.columns}

        # Perform feature selection with each model
        for model_name, model in models.items():
            logger.info(f"Feature selection with {model_name}...")

            # Fit model
            model.fit(X, y)

            # Get feature importances
            if model_name == 'LassoCV':
                importance = np.abs(model.coef_)
            else:
                importance = model.feature_importances_

            # Select features using mean importance threshold
            threshold = np.mean(importance)
            mask = importance > threshold
            model_selected = X.columns[mask].tolist()

            # Store selected features
            selected_features[model_name] = model_selected

            # Update feature votes
            for feature in model_selected:
                feature_votes[feature] += 1

            logger.info(f"{model_name} selected {len(model_selected)} features: {model_selected}")

        # Find consensus features (selected by majority of models)
        majority_threshold = len(models) / 2
        consensus_features = [feature for feature, votes in feature_votes.items()
                              if votes > majority_threshold]

        logger.info(f"Consensus features selected by majority of models: {consensus_features}")

        # Create feature selection visualization
        plt.figure(figsize=(12, 8))

        # Create a binary matrix of feature selection
        feature_matrix = np.zeros((len(X.columns), len(models)))
        for i, feature in enumerate(X.columns):
            for j, model_name in enumerate(models.keys()):
                if feature in selected_features[model_name]:
                    feature_matrix[i, j] = 1

        # Plot feature selection matrix
        plt.imshow(feature_matrix, cmap='Blues', aspect='auto')
        plt.yticks(range(len(X.columns)), X.columns)
        plt.xticks(range(len(models)), models.keys(), rotation=45)
        plt.colorbar(ticks=[0, 1], label='Feature Selected')
        plt.title("Feature Selection by Multiple Models")
        plt.tight_layout()
        plt.savefig(os.path.join(self.experiment_dir, 'multi_model_feature_selection.png'))
        plt.close()

        # Plot feature vote count
        plt.figure(figsize=(12, 8))
        features = list(feature_votes.keys())
        votes = list(feature_votes.values())

        # Sort by votes
        sorted_indices = np.argsort(votes)
        sorted_features = [features[i] for i in sorted_indices]
        sorted_votes = [votes[i] for i in sorted_indices]

        plt.barh(range(len(sorted_features)), sorted_votes, align='center')
        plt.yticks(range(len(sorted_features)), sorted_features)
        plt.axvline(x=majority_threshold, color='r', linestyle='--',
                    label=f'Majority threshold: {majority_threshold}')
        plt.xlabel('Number of models that selected feature')
        plt.title('Feature Selection Consensus')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.experiment_dir, 'feature_selection_consensus.png'))
        plt.close()

        return consensus_features

    def cluster_and_select_features(self, X, y, correlation_threshold=0.8, selection_method='importance'):
        """
        FIXED VERSION: Select features ensuring only one version per base feature.
        No more Log_TotalPb AND TotalPb together!
        """
        import re
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.feature_selection import mutual_info_regression

        logger.info("FIXING FIXED clustering: one version per base feature only")

        # Parse all features to understand their base and type
        feature_info = {}
        for feature in X.columns:
            if feature.startswith('Log_'):
                base = feature[4:]
                feature_info[feature] = {'base': base, 'type': 'log', 'priority': 2}
            elif feature.startswith('Log1p_'):
                base = feature[6:]
                feature_info[feature] = {'base': base, 'type': 'log1p', 'priority': 2}
            elif feature.startswith('Sqrt_'):
                base = feature[5:]
                feature_info[feature] = {'base': base, 'type': 'sqrt', 'priority': 3}
            elif feature.startswith('Squared_'):
                base = feature[8:]
                feature_info[feature] = {'base': base, 'type': 'squared', 'priority': 4}
            elif feature.startswith('Ratio_'):
                # Extract base features from ratio
                ratio_match = re.match(r'Ratio_(.+)_to_(.+)', feature)
                if ratio_match:
                    numerator, denominator = ratio_match.groups()
                    feature_info[feature] = {
                        'base': f"{numerator}_to_{denominator}",
                        'type': 'ratio',
                        'priority': 5,
                        'numerator': numerator,
                        'denominator': denominator
                    }
            else:
                feature_info[feature] = {'base': feature, 'type': 'original', 'priority': 1}

        # Calculate importance scores
        if selection_method == 'importance':
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            scores = dict(zip(X.columns, rf.feature_importances_))
        else:
            mi_scores = mutual_info_regression(X, y, random_state=42)
            scores = dict(zip(X.columns, mi_scores))

        # Group by base feature
        base_groups = {}
        ratio_features = []

        for feature, info in feature_info.items():
            if info['type'] == 'ratio':
                ratio_features.append((feature, scores[feature], info))
            else:
                base = info['base']
                if base not in base_groups:
                    base_groups[base] = []
                base_groups[base].append((feature, scores[feature], info))

        logger.info(f"Found {len(base_groups)} base feature groups")
        logger.info(f"Found {len(ratio_features)} ratio features")

        # Select best from each base group
        selected_features = []
        used_bases = set()

        # Sort base groups by their best feature's score
        base_group_scores = []
        for base, features in base_groups.items():
            best_score = max(score for _, score, _ in features)
            base_group_scores.append((base, best_score, features))

        base_group_scores.sort(key=lambda x: x[1], reverse=True)

        # Select from top base groups
        for base, best_score, features in base_group_scores[:10]:  # Limit to top 10 base features
            # Find best feature in this group
            best_feature = max(features, key=lambda x: x[1])
            selected_features.append(best_feature[0])
            used_bases.add(base)

            logger.info(
                f"Base '{base}': selected '{best_feature[0]}' (type: {best_feature[2]['type']}, score: {best_feature[1]:.4f})")

        # Handle ratios - only if they don't conflict with selected bases
        valid_ratios = []
        for ratio_feature, ratio_score, ratio_info in ratio_features:
            numerator = ratio_info.get('numerator', '')
            denominator = ratio_info.get('denominator', '')

            # Check if either base is already used
            if numerator not in used_bases and denominator not in used_bases:
                valid_ratios.append((ratio_feature, ratio_score))
            else:
                logger.info(f"WARNING  Skipping '{ratio_feature}' - conflicts with selected bases")

        # Add only the best ratio (if any)
        if valid_ratios:
            valid_ratios.sort(key=lambda x: x[1], reverse=True)
            best_ratio = valid_ratios[0]
            selected_features.append(best_ratio[0])
            logger.info(f"Added best ratio: '{best_ratio[0]}' (score: {best_ratio[1]:.4f})")

        logger.info(f"SUCCESS Final selection: {len(selected_features)} features with no duplicates")
        logger.info(f"Selected: {selected_features}")

        # Return format compatible with existing code
        feature_clusters = [[f] for f in selected_features]
        selection_details = [
            {
                'cluster': [feature],
                'selected': feature,
                'selection_criteria': 'no_duplicate_bases',
                'score': scores[feature]
            }
            for feature in selected_features
        ]

        return selected_features, feature_clusters, selection_details


def create_feature_correlation_heatmap(X, selected_features, experiment_dir,
                                       figsize=(12, 10), save_name='feature_correlation_heatmap.png'):
    """
    Create a correlation heatmap for selected features (standalone function).

    Parameters:
    -----------
    X : DataFrame
        Feature matrix containing all features
    selected_features : list
        List of feature names to include in heatmap
    experiment_dir : str
        Directory to save the plot
    figsize : tuple, default=(12, 10)
        Figure size for the heatmap
    save_name : str, default='feature_correlation_heatmap.png'
        Name of the saved plot file

    Returns:
    --------
    correlation_matrix : DataFrame
        Correlation matrix of the selected features
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import os

    if selected_features is None or len(selected_features) == 0:
        logger.error("No features provided for correlation heatmap")
        return None

    logger.info(f"Creating correlation heatmap for {len(selected_features)} selected features...")

    # Select only the specified features
    X_selected = X[selected_features]

    # Handle any remaining NaN values
    if X_selected.isnull().sum().sum() > 0:
        logger.warning("Found NaN values in selected features. Filling with column means for correlation calculation.")
        X_selected = X_selected.fillna(X_selected.mean())

    # Calculate correlation matrix
    correlation_matrix = X_selected.corr()

    # Create the heatmap
    plt.figure(figsize=figsize)

    # Create a mask for the upper triangle (optional - removes redundant info)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    # Generate the heatmap
    sns.heatmap(correlation_matrix,
                mask=mask,  # Comment this line if you want full matrix
                annot=True,
                cmap='RdBu_r',
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={"shrink": .8},
                annot_kws={'size': 8})

    plt.title(f'Feature Correlation Heatmap\n({len(selected_features)} Selected Features)',
              fontsize=14, fontweight='bold')
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Features', fontsize=12)

    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()

    # Save the plot
    if experiment_dir:
        os.makedirs(experiment_dir, exist_ok=True)
        save_path = os.path.join(experiment_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Correlation heatmap saved to: {save_path}")

    plt.show()
    plt.close()

    # Log high correlations
    logger.info("Analyzing feature correlations...")
    high_corr_pairs = []

    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) > 0.7:  # High correlation threshold
                feature1 = correlation_matrix.columns[i]
                feature2 = correlation_matrix.columns[j]
                high_corr_pairs.append((feature1, feature2, corr_value))

    if high_corr_pairs:
        logger.warning(f"Found {len(high_corr_pairs)} highly correlated feature pairs (|r| > 0.7):")
        for feature1, feature2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
            logger.warning(f"  {feature1} <-> {feature2}: r = {corr:.3f}")
    else:
        logger.info("No highly correlated feature pairs found (all |r| <= 0.7)")

    # Save correlation matrix to CSV
    if experiment_dir:
        csv_path = os.path.join(experiment_dir, 'feature_correlation_matrix.csv')
        correlation_matrix.to_csv(csv_path)
        logger.info(f"Correlation matrix saved to: {csv_path}")

    return correlation_matrix


def create_advanced_feature_analysis(X, y, selected_features, experiment_dir):
    """
    Create comprehensive feature analysis including correlation heatmap and additional plots (standalone function).

    Parameters:
    -----------
    X : DataFrame
        Feature matrix
    y : Series
        Target variable
    selected_features : list
        List of feature names to analyze
    experiment_dir : str
        Directory to save plots

    Returns:
    --------
    analysis_results : dict
        Dictionary containing analysis results
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import os
    from scipy.cluster.hierarchy import dendrogram, linkage
    from sklearn.feature_selection import mutual_info_regression

    if selected_features is None or len(selected_features) == 0:
        logger.error("No features provided for analysis")
        return None

    logger.info(f"Creating comprehensive feature analysis for {len(selected_features)} features...")

    # Ensure experiment directory exists
    os.makedirs(experiment_dir, exist_ok=True)

    X_selected = X[selected_features]

    # Handle NaN values
    if X_selected.isnull().sum().sum() > 0:
        X_selected = X_selected.fillna(X_selected.mean())

    # 1. Correlation heatmap (main function)
    correlation_matrix = create_feature_correlation_heatmap(
        X, selected_features=selected_features, experiment_dir=experiment_dir
    )

    # 2. Feature importance vs target correlation
    plt.figure(figsize=(12, 8))

    # Calculate correlations with target
    target_correlations = X_selected.corrwith(y)

    # Calculate mutual information
    mi_scores = mutual_info_regression(X_selected, y, random_state=42)
    mi_scores = pd.Series(mi_scores, index=X_selected.columns)

    # Create scatter plot
    plt.scatter(target_correlations, mi_scores, alpha=0.7, s=100)

    # Add feature names as labels
    for i, feature in enumerate(selected_features):
        plt.annotate(feature, (target_correlations[feature], mi_scores[feature]),
                     xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)

    plt.xlabel('Correlation with Target', fontsize=12)
    plt.ylabel('Mutual Information Score', fontsize=12)
    plt.title('Feature Importance: Correlation vs Mutual Information', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(experiment_dir, 'feature_importance_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Feature importance comparison saved to: {save_path}")
    plt.show()
    plt.close()

    # 3. Hierarchical clustering of features
    plt.figure(figsize=(15, 8))

    # Use absolute correlation for distance
    distance_matrix = 1 - np.abs(correlation_matrix)

    # Perform hierarchical clustering
    linkage_matrix = linkage(distance_matrix, method='ward')

    # Create dendrogram
    dendrogram(linkage_matrix, labels=selected_features, orientation='top')
    plt.title('Feature Hierarchical Clustering\n(Based on Correlation Distance)',
              fontsize=14, fontweight='bold')
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Distance', fontsize=12)
    plt.xticks(rotation=45, ha='right')

    save_path = os.path.join(experiment_dir, 'feature_clustering_dendrogram.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Feature clustering dendrogram saved to: {save_path}")
    plt.show()
    plt.close()

    # 4. Feature distribution plots
    n_features = len(selected_features)
    n_cols = min(4, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    if n_rows > 0:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.flatten()
        elif n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        for i, feature in enumerate(selected_features):
            if i < len(axes):
                axes[i].hist(X_selected[feature], bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{feature}', fontsize=10)
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(len(selected_features), len(axes)):
            axes[i].set_visible(False)

        plt.suptitle('Feature Distributions', fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(experiment_dir, 'feature_distributions.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature distributions saved to: {save_path}")
        plt.show()
        plt.close()

    # Compile analysis results
    analysis_results = {
        'correlation_matrix': correlation_matrix,
        'target_correlations': target_correlations,
        'mutual_info_scores': mi_scores,
        'high_correlation_pairs': [],
        'feature_statistics': X_selected.describe()
    }

    # Find high correlation pairs
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) > 0.7:
                feature1 = correlation_matrix.columns[i]
                feature2 = correlation_matrix.columns[j]
                analysis_results['high_correlation_pairs'].append({
                    'feature1': feature1,
                    'feature2': feature2,
                    'correlation': corr_value
                })

    logger.info("Comprehensive feature analysis completed")

    return analysis_results


def create_simple_correlation_heatmap(X, selected_features, experiment_dir,
                                      correlation_threshold=0.7, figsize=None):
    """
    Create a simple correlation heatmap with minimal dependencies (standalone function).

    Parameters:
    -----------
    X : DataFrame
        Feature matrix containing all features
    selected_features : list
        List of feature names to include in heatmap
    experiment_dir : str
        Directory to save the plot
    correlation_threshold : float, default=0.7
        Threshold for identifying high correlations
    figsize : tuple, optional
        Figure size. If None, will be calculated based on number of features

    Returns:
    --------
    correlation_matrix : DataFrame
        Correlation matrix of the selected features
    high_corr_pairs : list
        List of highly correlated feature pairs
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import os

    if selected_features is None or len(selected_features) == 0:
        logger.error("No features provided for correlation heatmap")
        return None, []

    logger.info(f"Creating simple correlation heatmap for {len(selected_features)} selected features...")

    # Select only the specified features
    X_selected = X[selected_features]

    # Handle any remaining NaN values
    if X_selected.isnull().sum().sum() > 0:
        logger.warning("Found NaN values in selected features. Filling with column means for correlation calculation.")
        X_selected = X_selected.fillna(X_selected.mean())

    # Calculate correlation matrix
    correlation_matrix = X_selected.corr()

    # Calculate figure size if not provided
    if figsize is None:
        n_features = len(selected_features)
        size = max(8, min(16, n_features * 0.8))
        figsize = (size, size)

    # Create the heatmap using matplotlib
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap manually
    im = ax.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

    # Set ticks and labels
    ax.set_xticks(range(len(selected_features)))
    ax.set_yticks(range(len(selected_features)))
    ax.set_xticklabels(selected_features, rotation=45, ha='right')
    ax.set_yticklabels(selected_features)

    # Add correlation values as text
    for i in range(len(selected_features)):
        for j in range(len(selected_features)):
            text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=8)

    # Add colorbar
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.title(f'Feature Correlation Heatmap\n({len(selected_features)} Selected Features)',
              fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save the plot
    if experiment_dir:
        os.makedirs(experiment_dir, exist_ok=True)
        save_path = os.path.join(experiment_dir, 'simple_feature_correlation_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Simple correlation heatmap saved to: {save_path}")

    plt.show()
    plt.close()

    # Find high correlations
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) > correlation_threshold:
                feature1 = correlation_matrix.columns[i]
                feature2 = correlation_matrix.columns[j]
                high_corr_pairs.append((feature1, feature2, corr_value))

    if high_corr_pairs:
        logger.warning(f"Found {len(high_corr_pairs)} highly correlated feature pairs (|r| > {correlation_threshold}):")
        for feature1, feature2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
            logger.warning(f"  {feature1} <-> {feature2}: r = {corr:.3f}")
    else:
        logger.info(f"No highly correlated feature pairs found (all |r| <= {correlation_threshold})")

    # Save correlation matrix to CSV
    if experiment_dir:
        csv_path = os.path.join(experiment_dir, 'simple_feature_correlation_matrix.csv')
        correlation_matrix.to_csv(csv_path)
        logger.info(f"Correlation matrix saved to: {csv_path}")

    return correlation_matrix, high_corr_pairs


