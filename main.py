import matplotlib
import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score


# Debug matplotlib import
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import custom modules
from data_management import DataLoader, DataValidator, DataSplitter
from preprocessing import PreprocessingPipeline
from feature_selection import FeatureSelector, create_feature_correlation_heatmap,create_advanced_feature_analysis, create_simple_correlation_heatmap
from feature_importance import FeatureImportanceVisualizer
from modeling import ModelTrainer, ModelEvaluator
from modeling_test import add_advanced_ensembles_to_pipeline
from partial_dependence import PartialDependenceAnalyzer
from utils import setup_logging, create_experiment_dir
from multi_criteria_selector import integrate_weighted_selector_in_main
from multi_criteria_selector import compare_multi_criteria_results as compare_weighted_results
from stratified_resampling_validation import retrain_with_explicit_control_enhanced, integrate_stratified_validation_in_retrain_function
from Advanced_validation_modelling import integrate_iterative_synthetic_validation_in_retrain_function, integrate_outlier_removal_into_retrain_results


# Set up logging
logger = setup_logging()


def debug_processed_data(X_train_processed, X_test_processed, stage_name):
    """
    Debug function to check processed data types.
    """
    logger.info(f"=== DEBUGGING {stage_name} ===")
    logger.info(f"X_train_processed shape: {X_train_processed.shape}")
    logger.info(f"X_train_processed dtypes:")

    # Check data types
    for col in X_train_processed.columns:
        dtype = X_train_processed[col].dtype
        logger.info(f"  {col}: {dtype}")

        # If it's object type, show sample values
        if dtype == 'object':
            unique_vals = X_train_processed[col].dropna().unique()
            logger.warning(f"    WARNING Object column '{col}' has values: {list(unique_vals[:5])}")

    # Check for non-numeric columns
    numeric_cols = X_train_processed.select_dtypes(include=[np.number]).columns
    non_numeric_cols = X_train_processed.select_dtypes(exclude=[np.number]).columns

    logger.info(f"Numeric columns: {len(numeric_cols)}")
    logger.info(f"Non-numeric columns: {len(non_numeric_cols)}")

    if len(non_numeric_cols) > 0:
        logger.error(f"ERROR Found non-numeric columns in processed data: {list(non_numeric_cols)}")
        for col in non_numeric_cols:
            sample_vals = X_train_processed[col].dropna().unique()[:3]
            logger.error(f"  {col} sample values: {list(sample_vals)}")
    else:
        logger.info("SUCCESS All columns are numeric")


def create_bootstrap_missing_data(X_train_processed, X_test_processed, y_train_processed, y_test_processed,
                                  missing_percentages=[10, 20, 30], experiment_dir=None):
    """
    Create bootstrap datasets with artificial missing data from already-preprocessed data.
    """
    logger.info("Creating bootstrap datasets from preprocessed data...")

    bootstrap_datasets = {}
    all_features = X_train_processed.columns.tolist()

    for missing_pct in missing_percentages:
        logger.info(f"  Creating {missing_pct}% missing data scenario...")

        # Set seed for reproducibility
        np.random.seed(42 + missing_pct)

        # Calculate number of missing values
        n_train_missing = int(len(X_train_processed) * missing_pct / 100)
        n_test_missing = int(len(X_test_processed) * missing_pct / 100)

        # Randomly select rows and features to make missing
        train_missing_rows = np.random.choice(len(X_train_processed), n_train_missing, replace=False)
        test_missing_rows = np.random.choice(len(X_test_processed), n_test_missing, replace=False)

        # Select random features to make missing (about 25% of features)
        n_features_missing = max(1, len(all_features) // 4)
        missing_features = np.random.choice(all_features, n_features_missing, replace=False)

        # Create copies with missing data
        X_train_missing = X_train_processed.copy()
        X_test_missing = X_test_processed.copy()

        # Introduce missing values
        for feature in missing_features:
            X_train_missing.loc[X_train_missing.index[train_missing_rows], feature] = np.nan
            X_test_missing.loc[X_test_missing.index[test_missing_rows], feature] = np.nan

        # Store the bootstrap dataset
        scenario_name = f'{missing_pct}pct_missing'
        bootstrap_datasets[scenario_name] = {
            'X_train': X_train_missing,
            'X_test': X_test_missing,
            'y_train': y_train_processed,
            'y_test': y_test_processed,
            'missing_pct': missing_pct,
            'train_missing_count': n_train_missing,
            'test_missing_count': n_test_missing,
            'missing_features': list(missing_features)
        }

        logger.info(
            f"    Created scenario: {n_train_missing} train + {n_test_missing} test rows missing in {n_features_missing} features")

    return bootstrap_datasets


def save_bootstrap_metrics(bootstrap_results, imputation_method, experiment_dir):
    """
    Save bootstrap experiment metrics to files.
    """
    bootstrap_dir = os.path.join(experiment_dir, 'bootstrap_metrics')
    os.makedirs(bootstrap_dir, exist_ok=True)

    # Create detailed results DataFrame
    bootstrap_data = []

    for scenario_name, results in bootstrap_results.items():
        if 'error' not in results:
            bootstrap_data.append({
                'scenario': scenario_name,
                'imputation_method': imputation_method,
                'missing_pct': results.get('missing_pct', 0),
                'train_r2': results.get('train_r2', 0),
                'test_r2': results.get('test_r2', 0),
                'cv_r2_mean': results.get('cv_r2_mean', 0),
                'cv_r2_std': results.get('cv_r2_std', 0),
                'features_selected': results.get('features_selected', 0),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        else:
            bootstrap_data.append({
                'scenario': scenario_name,
                'imputation_method': imputation_method,
                'missing_pct': 0,
                'train_r2': None,
                'test_r2': None,
                'cv_r2_mean': None,
                'cv_r2_std': None,
                'features_selected': None,
                'error': results['error'],
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

    if bootstrap_data:
        # Save detailed results
        bootstrap_df = pd.DataFrame(bootstrap_data)
        csv_path = os.path.join(bootstrap_dir, f'bootstrap_results_{imputation_method}.csv')
        bootstrap_df.to_csv(csv_path, index=False)
        logger.info(f"Bootstrap results saved to {csv_path}")

        # Save summary statistics
        successful_results = bootstrap_df[bootstrap_df['test_r2'].notna()]
        if len(successful_results) > 0:
            summary_stats = {
                'imputation_method': imputation_method,
                'total_scenarios': len(bootstrap_df),
                'successful_scenarios': len(successful_results),
                'failed_scenarios': len(bootstrap_df) - len(successful_results),
                'mean_test_r2': successful_results['test_r2'].mean(),
                'std_test_r2': successful_results['test_r2'].std(),
                'mean_cv_r2': successful_results['cv_r2_mean'].mean(),
                'std_cv_r2': successful_results['cv_r2_mean'].std(),
                'best_scenario': successful_results.loc[successful_results['test_r2'].idxmax(), 'scenario'],
                'best_test_r2': successful_results['test_r2'].max(),
                'worst_scenario': successful_results.loc[successful_results['test_r2'].idxmin(), 'scenario'],
                'worst_test_r2': successful_results['test_r2'].min(),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            # Save summary
            summary_path = os.path.join(bootstrap_dir, f'bootstrap_summary_{imputation_method}.json')
            with open(summary_path, 'w') as f:
                json.dump(summary_stats, f, indent=4)
            logger.info(f"Bootstrap summary saved to {summary_path}")

            # Create visualization
            create_bootstrap_visualization(successful_results, imputation_method, bootstrap_dir)

        return bootstrap_df
    else:
        logger.warning("No bootstrap data to save")
        return None

def compare_cv_results(simple_results, iterative_results, experiment_dir):
    """
    Compare the cross-validation results from different imputation methods.
    """
    logger.info("Comparing cross-validation results between imputation methods...")

    # Create comparison directory
    comparison_dir = os.path.join(experiment_dir, 'cv_imputation_comparison')
    os.makedirs(comparison_dir, exist_ok=True)

    # Extract results
    simple_eval = simple_results['evaluation_results']
    iterative_eval = iterative_results['evaluation_results']

    # Find models with CV results
    models = [model for model in simple_eval.keys() if
              'cv_test_r2_mean' in simple_eval[model] and 'cv_test_r2_mean' in iterative_eval[model]]

    if not models:
        logger.warning("No models with cross-validation results found for comparison")
        return None

    # Extract CV metrics
    simple_cv_r2 = [simple_eval[model]['cv_test_r2_mean'] for model in models]
    simple_cv_r2_std = [simple_eval[model]['cv_test_r2_std'] for model in models]
    iterative_cv_r2 = [iterative_eval[model]['cv_test_r2_mean'] for model in models]
    iterative_cv_r2_std = [iterative_eval[model]['cv_test_r2_std'] for model in models]

    # Create comparison visualization
    plt.figure(figsize=(14, 8))
    x = np.arange(len(models))
    width = 0.35

    plt.bar(x - width / 2, simple_cv_r2, width, label='Simple Imputation CV R²', color='blue', alpha=0.7)
    plt.bar(x + width / 2, iterative_cv_r2, width, label='Iterative Imputation CV R²', color='green', alpha=0.7)

    plt.errorbar(x - width / 2, simple_cv_r2, yerr=simple_cv_r2_std, fmt='none', capsize=5, color='black', alpha=0.5)
    plt.errorbar(x + width / 2, iterative_cv_r2, yerr=iterative_cv_r2_std, fmt='none', capsize=5, color='black',
                 alpha=0.5)

    plt.xlabel('Models',fontsize=14)
    plt.ylabel('Cross-Validation Test R² Score', fontsize=14)
    plt.title('Model Performance Comparison by Imputation Method (Cross-Validated)', fontsize=16)
    plt.xticks(x, models, rotation=45, fontsize=10)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add values on bars
    for i, v in enumerate(simple_cv_r2):
        plt.text(i - width / 2, v + 0.01, f'{v:.3f}+/-{simple_cv_r2_std[i]:.3f}', ha='center', fontsize=8)
    for i, v in enumerate(iterative_cv_r2):
        plt.text(i + width / 2, v + 0.01, f'{v:.3f}+/-{iterative_cv_r2_std[i]:.3f}', ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'cv_imputation_comparison.png'))
    plt.close()

    # Compare bootstrap results if available
    simple_bootstrap = simple_results.get('bootstrap_results', {})
    iterative_bootstrap = iterative_results.get('bootstrap_results', {})

    if simple_bootstrap and iterative_bootstrap:
        logger.info("Comparing bootstrap results between imputation methods...")

        bootstrap_comparison = []
        for scenario in simple_bootstrap.keys():
            if scenario in iterative_bootstrap and 'error' not in simple_bootstrap[scenario] and 'error' not in \
                    iterative_bootstrap[scenario]:
                simple_r2 = simple_bootstrap[scenario].get('test_r2', 0)
                iterative_r2 = iterative_bootstrap[scenario].get('test_r2', 0)

                bootstrap_comparison.append({
                    'scenario': scenario,
                    'missing_pct': simple_bootstrap[scenario].get('missing_pct', 0),
                    'simple_r2': simple_r2,
                    'iterative_r2': iterative_r2,
                    'improvement': iterative_r2 - simple_r2
                })

        if bootstrap_comparison:
            bootstrap_df = pd.DataFrame(bootstrap_comparison)
            bootstrap_df.to_csv(os.path.join(comparison_dir, 'bootstrap_comparison.csv'), index=False)

            logger.info("Bootstrap Results Summary:")
            for _, row in bootstrap_df.iterrows():
                logger.info(
                    f"  {row['scenario']}: Simple={row['simple_r2']:.4f}, Iterative={row['iterative_r2']:.4f}, Improvement={row['improvement']:.4f}")

    # Find best models
    best_simple_cv_model = max([(model, simple_eval[model]['cv_test_r2_mean']) for model in models], key=lambda x: x[1])
    best_iterative_cv_model = max([(model, iterative_eval[model]['cv_test_r2_mean']) for model in models],
                                  key=lambda x: x[1])

    # Determine overall best
    if best_simple_cv_model[1] > best_iterative_cv_model[1]:
        best_overall = (best_simple_cv_model[0], 'Simple', best_simple_cv_model[1])
    else:
        best_overall = (best_iterative_cv_model[0], 'Iterative', best_iterative_cv_model[1])

    logger.info(
        f"Best overall model (CV): {best_overall[0]} with {best_overall[1]} imputation (CV R² = {best_overall[2]:.4f})")

    return best_overall


def generate_partial_dependence_for_best_cv_model(best_results, best_model_name, experiment_dir):
    """
    Generate partial dependence plots for the best CV model.
    """
    # Extract necessary components
    models = best_results['models']
    model_key = f"{best_model_name}_optimized" if f"{best_model_name}_optimized" in models else best_model_name
    best_model = models[model_key]

    X_train_selected = best_results['X_train_selected']
    y_train_processed = best_results['y_train_processed']
    selected_features = best_results.get('selected_features', X_train_selected.columns.tolist())

    logger.info(f"Generating partial dependence plots for best CV model: {best_model_name}")
    logger.info(f"ANALYZING Model was trained on {len(selected_features)} features: {selected_features}")
    logger.info(f"ANALYZING X_train_selected has {len(X_train_selected.columns)} features: {list(X_train_selected.columns)}")

    # SUCCESS VALIDATION: Ensure feature consistency
    if set(selected_features) != set(X_train_selected.columns):
        logger.warning("WARNING Feature mismatch detected! Aligning features...")
        logger.warning(f"Selected features: {selected_features}")
        logger.warning(f"X_train columns: {list(X_train_selected.columns)}")

        # Use the intersection of features that exist in both
        common_features = list(set(selected_features) & set(X_train_selected.columns))
        X_train_selected = X_train_selected[common_features]
        logger.info(f"SUCCESS Using {len(common_features)} common features for PDP analysis")

    # SUCCESS FINAL CHECK: Ensure the model expects these exact features
    try:
        # Test prediction to ensure feature compatibility
        test_prediction = best_model.predict(X_train_selected.iloc[:1])
        logger.info(f"SUCCESS Model prediction test successful with {X_train_selected.shape[1]} features")
    except Exception as e:
        logger.error(f"ERROR Model prediction test failed: {str(e)}")
        logger.error("This indicates a serious feature mismatch issue")
        return


    logger.info(f"Generating partial dependence plots for best CV model: {best_model_name}")

    # Create PDP directory
    pdp_dir = os.path.join(experiment_dir, 'best_cv_model_pdp')
    os.makedirs(pdp_dir, exist_ok=True)

    # Initialize the partial dependence analyzer
    pdp_analyzer = PartialDependenceAnalyzer(pdp_dir)

    # Generate partial dependence plots
    pdp_analyzer.generate_single_feature_pdp(best_model, X_train_selected, n_features=10)
    pdp_analyzer.generate_feature_grid_pdp(best_model, X_train_selected, n_features=6)

    # Look for TotalPb feature for interaction analysis
    total_pb_feature = None
    for feature in X_train_selected.columns:
        if 'totalpb' in feature.lower() or ('total' in feature.lower() and 'pb' in feature.lower()):
            total_pb_feature = feature
            break

    if total_pb_feature:
        logger.info(f"Found TotalPb feature: {total_pb_feature}. Analyzing interactions.")
        pdp_analyzer.generate_2d_interaction_pdp(
            best_model, X_train_selected, top_n=5, total_pb_feature=total_pb_feature
        )
        pdp_analyzer.analyze_totalPb_interactions(
            best_model, X_train_selected, y_train_processed, total_pb_feature=total_pb_feature
        )
    else:
        logger.info("TotalPb feature not found in selected features. Skipping TotalPb interaction analysis.")

    # Generate ICE plots
    pdp_analyzer.generate_ice_plots(best_model, X_train_selected, n_features=5)

    logger.info("Partial dependence analysis completed")


def save_combined_bootstrap_metrics(simple_results, iterative_results, experiment_dir):
    """
    Save combined bootstrap metrics comparing both imputation methods.
    """
    bootstrap_dir = os.path.join(experiment_dir, 'bootstrap_metrics')
    os.makedirs(bootstrap_dir, exist_ok=True)

    # Combine all bootstrap results
    all_bootstrap_data = []

    # Process simple imputation results
    simple_bootstrap = simple_results.get('bootstrap_results', {})
    for scenario_name, results in simple_bootstrap.items():
        if 'error' not in results:
            all_bootstrap_data.append({
                'scenario': scenario_name,
                'imputation_method': 'simple',
                'missing_pct': results.get('missing_pct', 0),
                'train_r2': results.get('train_r2', 0),
                'test_r2': results.get('test_r2', 0),
                'cv_r2_mean': results.get('cv_r2_mean', 0),
                'cv_r2_std': results.get('cv_r2_std', 0),
                'features_selected': results.get('features_selected', 0),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

    # Process iterative imputation results
    iterative_bootstrap = iterative_results.get('bootstrap_results', {})
    for scenario_name, results in iterative_bootstrap.items():
        if 'error' not in results:
            all_bootstrap_data.append({
                'scenario': scenario_name,
                'imputation_method': 'iterative',
                'missing_pct': results.get('missing_pct', 0),
                'train_r2': results.get('train_r2', 0),
                'test_r2': results.get('test_r2', 0),
                'cv_r2_mean': results.get('cv_r2_mean', 0),
                'cv_r2_std': results.get('cv_r2_std', 0),
                'features_selected': results.get('features_selected', 0),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

    if all_bootstrap_data:
        # Save combined results
        combined_df = pd.DataFrame(all_bootstrap_data)
        csv_path = os.path.join(bootstrap_dir, 'combined_bootstrap_results.csv')
        combined_df.to_csv(csv_path, index=False)
        logger.info(f"Combined bootstrap results saved to {csv_path}")

        # Create comparison visualization
        create_combined_bootstrap_visualization(combined_df, bootstrap_dir)

        return combined_df
    else:
        logger.warning("No bootstrap data to combine")
        return None


def create_combined_bootstrap_visualization(combined_df, bootstrap_dir):
    """
    Create combined visualization comparing both imputation methods.
    """
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Performance comparison by missing data percentage
        ax1 = axes[0, 0]

        for method in ['simple', 'iterative']:
            method_data = combined_df[combined_df['imputation_method'] == method]
            missing_pcts = sorted(method_data['missing_pct'].unique())

            avg_r2 = []
            std_r2 = []

            for pct in missing_pcts:
                subset = method_data[method_data['missing_pct'] == pct]
                avg_r2.append(subset['test_r2'].mean())
                std_r2.append(subset['test_r2'].std())

            ax1.errorbar(missing_pcts, avg_r2, yerr=std_r2,
                         marker='o', label=f'{method.title()} Imputation',
                         linewidth=2, markersize=8, capsize=5)

        ax1.set_xlabel('Missing Data Percentage (%)')
        ax1.set_ylabel('Test R² Score')
        ax1.set_title('Imputation Method Comparison: Performance vs Missing Data')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Direct comparison boxplot
        ax2 = axes[0, 1]

        simple_scores = combined_df[combined_df['imputation_method'] == 'simple']['test_r2']
        iterative_scores = combined_df[combined_df['imputation_method'] == 'iterative']['test_r2']

        ax2.boxplot([simple_scores, iterative_scores], labels=['Simple', 'Iterative'])
        ax2.set_ylabel('Test R² Score')
        ax2.set_title('Overall Performance Distribution Comparison')
        ax2.grid(True, alpha=0.3)

        # Plot 3: CV vs Test R² scatter by method
        ax3 = axes[1, 0]

        for method, color in [('simple', 'blue'), ('iterative', 'green')]:
            method_data = combined_df[combined_df['imputation_method'] == method]
            ax3.scatter(method_data['cv_r2_mean'], method_data['test_r2'],
                        alpha=0.7, s=60, color=color, label=f'{method.title()} Imputation')

        # Add diagonal line
        min_val = combined_df[['cv_r2_mean', 'test_r2']].min().min()
        max_val = combined_df[['cv_r2_mean', 'test_r2']].max().max()
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect Agreement')

        ax3.set_xlabel('CV R² Score')
        ax3.set_ylabel('Test R² Score')
        ax3.set_title('CV vs Test R² by Imputation Method')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Improvement analysis
        ax4 = axes[1, 1]

        # Calculate improvements for each scenario
        improvements = []
        scenarios = []

        for scenario in combined_df['scenario'].unique():
            simple_score = combined_df[(combined_df['scenario'] == scenario) &
                                       (combined_df['imputation_method'] == 'simple')]['test_r2']
            iterative_score = combined_df[(combined_df['scenario'] == scenario) &
                                          (combined_df['imputation_method'] == 'iterative')]['test_r2']

            if len(simple_score) > 0 and len(iterative_score) > 0:
                improvement = iterative_score.iloc[0] - simple_score.iloc[0]
                improvements.append(improvement)
                scenarios.append(scenario)

        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        bars = ax4.bar(range(len(improvements)), improvements, color=colors, alpha=0.7)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.set_xlabel('Bootstrap Scenarios')
        ax4.set_ylabel('Improvement (Iterative - Simple)')
        ax4.set_title('Iterative vs Simple Improvement by Scenario')
        ax4.set_xticks(range(len(scenarios)))
        ax4.set_xticklabels(scenarios, rotation=45)
        ax4.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, improvement in zip(bars, improvements):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.001 * (1 if height > 0 else -1),
                     f'{improvement:.3f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(bootstrap_dir, 'combined_bootstrap_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Combined bootstrap visualization saved to {plot_path}")

    except Exception as e:
        logger.error(f"Error creating combined bootstrap visualization: {str(e)}")
def manual_categorical_handling(X_train, X_test, categorical_features, numeric_features):
    """
    Manual fallback for categorical data handling when preprocessing pipeline fails.
    """
    logger.info("FIXING Applying manual categorical data handling...")

    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler

    # Separate numeric and categorical data
    X_train_numeric = X_train[numeric_features].copy()
    X_test_numeric = X_test[numeric_features].copy()

    processed_dfs = [X_train_numeric, X_test_numeric]

    if categorical_features:
        X_train_categorical = X_train[categorical_features].copy()
        X_test_categorical = X_test[categorical_features].copy()

        # Apply one-hot encoding
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')

        # Fit on training data only
        X_train_cat_encoded = ohe.fit_transform(X_train_categorical)
        X_test_cat_encoded = ohe.transform(X_test_categorical)

        # Create column names for encoded features
        cat_feature_names = ohe.get_feature_names_out(categorical_features)

        # Convert to DataFrames
        X_train_cat_df = pd.DataFrame(
            X_train_cat_encoded,
            columns=cat_feature_names,
            index=X_train.index
        )
        X_test_cat_df = pd.DataFrame(
            X_test_cat_encoded,
            columns=cat_feature_names,
            index=X_test.index
        )

        processed_dfs.extend([X_train_cat_df, X_test_cat_df])

        logger.info(
            f"SUCCESS One-hot encoded {len(categorical_features)} categorical features into {len(cat_feature_names)} features")

    # Combine numeric and encoded categorical features
    X_train_combined = pd.concat([processed_dfs[0], processed_dfs[2]] if len(processed_dfs) > 2 else [processed_dfs[0]],
                                 axis=1)
    X_test_combined = pd.concat([processed_dfs[1], processed_dfs[3]] if len(processed_dfs) > 2 else [processed_dfs[1]],
                                axis=1)

    # Apply basic scaling to numeric features only
    scaler = StandardScaler()
    X_train_combined[numeric_features] = scaler.fit_transform(X_train_combined[numeric_features])
    X_test_combined[numeric_features] = scaler.transform(X_test_combined[numeric_features])

    logger.info(f"SUCCESS Manual processing complete. Final shape: {X_train_combined.shape}")

    return X_train_combined, X_test_combined


def validate_processed_data(X_train_processed, X_test_processed, stage_name):
    """
    Enhanced validation function for processed data.
    """
    logger.info(f"=== ENHANCED VALIDATION: {stage_name} ===")
    logger.info(f"Training data shape: {X_train_processed.shape}")
    logger.info(f"Test data shape: {X_test_processed.shape}")

    # Check data types
    numeric_cols = X_train_processed.select_dtypes(include=[np.number]).columns
    non_numeric_cols = X_train_processed.select_dtypes(exclude=[np.number]).columns

    logger.info(f"SUCCESS Numeric columns: {len(numeric_cols)}")
    logger.info(f"{'ERROR' if len(non_numeric_cols) > 0 else 'SUCCESS'} Non-numeric columns: {len(non_numeric_cols)}")

    if len(non_numeric_cols) > 0:
        logger.error(f"Non-numeric columns found: {list(non_numeric_cols)}")
        for col in non_numeric_cols[:3]:  # Show first 3 as examples
            sample_vals = X_train_processed[col].dropna().unique()[:3]
            logger.error(f"  '{col}' examples: {list(sample_vals)}")

    # Check for NaN values
    train_nan_count = X_train_processed.isna().sum().sum()
    test_nan_count = X_test_processed.isna().sum().sum()
    logger.info(f"{'ERROR' if train_nan_count > 0 else 'SUCCESS'} Training NaN values: {train_nan_count}")
    logger.info(f"{'ERROR' if test_nan_count > 0 else 'SUCCESS'} Test NaN values: {test_nan_count}")

    # Check for infinite values (only numeric columns)
    if len(numeric_cols) > 0:
        train_inf_count = np.isinf(X_train_processed[numeric_cols]).sum().sum()
        test_inf_count = np.isinf(X_test_processed[numeric_cols]).sum().sum()
        logger.info(f"{'WARNING' if train_inf_count > 0 else 'SUCCESS'} Training infinite values: {train_inf_count}")
        logger.info(f"{'WARNING' if test_inf_count > 0 else 'SUCCESS'} Test infinite values: {test_inf_count}")

    # Check feature consistency between train and test
    train_cols = set(X_train_processed.columns)
    test_cols = set(X_test_processed.columns)
    if train_cols != test_cols:
        logger.error(f"ERROR Feature mismatch between train and test!")
        logger.error(f"  Only in train: {train_cols - test_cols}")
        logger.error(f"  Only in test: {test_cols - train_cols}")
    else:
        logger.info("SUCCESS Feature consistency between train and test")

    logger.info("=" * 50)


def create_bootstrap_visualization(bootstrap_df, imputation_method, bootstrap_dir):
    """
    Create visualizations for bootstrap results.
    """
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Performance vs Missing Data Percentage
        ax1 = axes[0, 0]
        missing_pcts = bootstrap_df['missing_pct'].unique()
        missing_pcts = missing_pcts[missing_pcts > 0]  # Exclude 0% missing if present

        test_r2_by_missing = []
        cv_r2_by_missing = []
        cv_std_by_missing = []

        for pct in sorted(missing_pcts):
            subset = bootstrap_df[bootstrap_df['missing_pct'] == pct]
            test_r2_by_missing.append(subset['test_r2'].mean())
            cv_r2_by_missing.append(subset['cv_r2_mean'].mean())
            cv_std_by_missing.append(subset['cv_r2_std'].mean())

        ax1.plot(sorted(missing_pcts), test_r2_by_missing, 'o-', label='Test R²', linewidth=2, markersize=8)
        ax1.errorbar(sorted(missing_pcts), cv_r2_by_missing, yerr=cv_std_by_missing,
                     fmt='s-', label='CV R² (+/-std)', linewidth=2, markersize=8, capsize=5)
        ax1.set_xlabel('Missing Data Percentage (%)')
        ax1.set_ylabel('R² Score')
        ax1.set_title(f'{imputation_method.title()} Imputation: Performance vs Missing Data')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Test R² Distribution
        ax2 = axes[0, 1]
        ax2.hist(bootstrap_df['test_r2'], bins=10, alpha=0.7, edgecolor='black')
        ax2.axvline(bootstrap_df['test_r2'].mean(), color='red', linestyle='--',
                    label=f'Mean: {bootstrap_df["test_r2"].mean():.4f}')
        ax2.set_xlabel('Test R² Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'{imputation_method.title()} Imputation: Test R² Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: CV R² vs Test R²
        ax3 = axes[1, 0]
        ax3.scatter(bootstrap_df['cv_r2_mean'], bootstrap_df['test_r2'], alpha=0.7, s=50)

        # Add diagonal line
        min_val = min(bootstrap_df['cv_r2_mean'].min(), bootstrap_df['test_r2'].min())
        max_val = max(bootstrap_df['cv_r2_mean'].max(), bootstrap_df['test_r2'].max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect Agreement')

        ax3.set_xlabel('CV R² Score')
        ax3.set_ylabel('Test R² Score')
        ax3.set_title(f'{imputation_method.title()} Imputation: CV vs Test R²')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Features Selected vs Performance
        ax4 = axes[1, 1]
        ax4.scatter(bootstrap_df['features_selected'], bootstrap_df['test_r2'], alpha=0.7, s=50)
        ax4.set_xlabel('Number of Features Selected')
        ax4.set_ylabel('Test R² Score')
        ax4.set_title(f'{imputation_method.title()} Imputation: Features vs Performance')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(bootstrap_dir, f'bootstrap_analysis_{imputation_method}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Bootstrap visualization saved to {plot_path}")

    except Exception as e:
        logger.error(f"Error creating bootstrap visualization: {str(e)}")


def run_bootstrap_experiments(X_train_processed, X_test_processed, y_train_processed, y_test_processed,
                              imputation_method, imputer_dir):
    """
    Run bootstrap experiments after preprocessing is complete.
    UPDATED with debugging and metrics saving.
    """
    logger.info("=" * 50)
    logger.info(f"Running bootstrap experiments with {imputation_method} imputation...")

    # DEBUG: Check the input data types
    debug_processed_data(X_train_processed, X_test_processed, "BOOTSTRAP INPUT")

    # If we find non-numeric data, force convert to numeric
    numeric_train_cols = X_train_processed.select_dtypes(include=[np.number]).columns
    non_numeric_train_cols = X_train_processed.select_dtypes(exclude=[np.number]).columns

    if len(non_numeric_train_cols) > 0:
        logger.error(f"ERROR PROBLEM: Non-numeric columns found in preprocessed data: {list(non_numeric_train_cols)}")
        logger.error("This suggests the preprocessing pipeline didn't handle categorical data properly")

        # Emergency fix: Drop non-numeric columns for bootstrap
        logger.warning("FIXING EMERGENCY FIX: Dropping non-numeric columns for bootstrap experiments")
        X_train_processed = X_train_processed[numeric_train_cols]
        X_test_processed = X_test_processed[numeric_train_cols]


        logger.info(f"After dropping non-numeric columns: {X_train_processed.shape[1]} features remaining")

    bootstrap_results = {}

    # Create bootstrap datasets from the preprocessed data
    bootstrap_datasets = create_bootstrap_missing_data(
        X_train_processed, X_test_processed,
        y_train_processed, y_test_processed,
        missing_percentages=[10, 20, 30],
        experiment_dir=imputer_dir
    )

    # Test each bootstrap scenario with the current imputation method
    for scenario_name, bootstrap_data in bootstrap_datasets.items():
        logger.info(f"Testing {scenario_name} with {imputation_method} imputation...")

        try:
            # Use the SAME preprocessing pipeline to handle the missing data
            X_train_bootstrap = bootstrap_data['X_train']
            X_test_bootstrap = bootstrap_data['X_test']

            # DEBUG: Check bootstrap data types
            logger.info(f"Bootstrap data types for {scenario_name}:")
            non_numeric_bootstrap = X_train_bootstrap.select_dtypes(exclude=[np.number]).columns
            if len(non_numeric_bootstrap) > 0:
                logger.error(f"ERROR Non-numeric columns in bootstrap data: {list(non_numeric_bootstrap)}")
                # Drop them
                numeric_bootstrap_cols = X_train_bootstrap.select_dtypes(include=[np.number]).columns
                X_train_bootstrap = X_train_bootstrap[numeric_bootstrap_cols]
                X_test_bootstrap = X_test_bootstrap[numeric_bootstrap_cols]
                logger.warning(f"FIXING Fixed: Using only {len(numeric_bootstrap_cols)} numeric columns")

            # Check if there are missing values to impute
            train_missing = X_train_bootstrap.isna().sum().sum()
            test_missing = X_test_bootstrap.isna().sum().sum()

            if train_missing > 0 or test_missing > 0:
                logger.info(f"  Imputing {train_missing} train + {test_missing} test missing values...")

                # Create a new imputer of the same type
                if imputation_method == 'simple':
                    from sklearn.impute import SimpleImputer
                    bootstrap_imputer = SimpleImputer(strategy='median')
                else:  # iterative
                    from sklearn.experimental import enable_iterative_imputer
                    from sklearn.impute import IterativeImputer
                    bootstrap_imputer = IterativeImputer(random_state=42, max_iter=10)

                # SAFETY CHECK: Ensure data is numeric before imputation
                if not all(X_train_bootstrap.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
                    logger.error("ERROR CRITICAL: Non-numeric data passed to imputer!")
                    logger.error(f"Data types: {X_train_bootstrap.dtypes}")
                    logger.error("Skipping this bootstrap scenario")
                    bootstrap_results[scenario_name] = {'error': 'Non-numeric data in bootstrap'}
                    continue

                # Fit imputer on bootstrap training data only
                bootstrap_imputer.fit(X_train_bootstrap)

                # Transform both sets
                X_train_bootstrap_imputed = pd.DataFrame(
                    bootstrap_imputer.transform(X_train_bootstrap),
                    columns=X_train_bootstrap.columns,
                    index=X_train_bootstrap.index
                )
                X_test_bootstrap_imputed = pd.DataFrame(
                    bootstrap_imputer.transform(X_test_bootstrap),
                    columns=X_test_bootstrap.columns,
                    index=X_test_bootstrap.index
                )
            else:
                X_train_bootstrap_imputed = X_train_bootstrap
                X_test_bootstrap_imputed = X_test_bootstrap

            # Quick evaluation with a simple model
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import r2_score

            # Simple feature selection (top 15 features by variance)
            from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
            selector = SelectKBest(f_regression, k=min(15, X_train_bootstrap_imputed.shape[1]))
            X_train_selected = selector.fit_transform(X_train_bootstrap_imputed, bootstrap_data['y_train'])
            X_test_selected = selector.transform(X_test_bootstrap_imputed)

            # Train and evaluate
            model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            model.fit(X_train_selected, bootstrap_data['y_train'])

            # Calculate metrics
            train_pred = model.predict(X_train_selected)
            test_pred = model.predict(X_test_selected)

            train_r2 = r2_score(bootstrap_data['y_train'], train_pred)
            test_r2 = r2_score(bootstrap_data['y_test'], test_pred)

            # Cross-validation
            cv_scores = cross_val_score(model, X_train_selected, bootstrap_data['y_train'], cv=3, scoring='r2')

            bootstrap_results[scenario_name] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'missing_pct': bootstrap_data['missing_pct'],
                'features_selected': X_train_selected.shape[1],
                'imputation_method': imputation_method
            }

            logger.info(
                f"  SUCCESS {scenario_name}: Test R²={test_r2:.4f}, CV R²={cv_scores.mean():.4f}+/-{cv_scores.std():.4f}")

        except Exception as e:
            logger.error(f"  ERROR Error in {scenario_name}: {str(e)}")
            import traceback
            logger.error(f"  Full traceback: {traceback.format_exc()}")
            bootstrap_results[scenario_name] = {'error': str(e)}

    return bootstrap_results


def run_pipeline_with_enhanced_model_selection(imputation_method, experiment_dir, X_train, X_test, y_train, y_test,
                                        enable_bootstrap=True,
                                        enable_feature_engineering=True,
                                        enable_knn_augmentation=False,  # NEW PARAMETER
                                        knn_expansion_factor=1.5,  # NEW PARAMETER
                                        knn_n_neighbors=5,  # NEW PARAMETER
                                        knn_noise_level=0.05,  # NEW PARAMETER
                                        knn_synthesis_method='knn'
                                        ):
    """
    Enhanced pipeline with multi-criteria model selection.
    """
    # Create subdirectory for this imputation method
    imputer_dir = os.path.join(experiment_dir, f'imputer_{imputation_method}_cv_improved')
    os.makedirs(imputer_dir, exist_ok=True)
    logger.info(f"Running pipeline with {imputation_method} imputation method in {imputer_dir}")
    logger.info(f"Feature engineering: {'ENABLED' if enable_feature_engineering else 'DISABLED'}")
    logger.info("=" * 50)

    # PREPROCESSING LAYER
    logger.info(f"Preprocessing data with {imputation_method} imputer...")
    logger.info(f"Enhanced preprocessing with categorical data protection...")

    # STEP 1: Detailed feature type analysis
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    logger.info(f"DATA Feature Analysis:")
    logger.info(f"  Total features: {len(X_train.columns)}")
    logger.info(f"  Numeric features ({len(numeric_features)}): {numeric_features}")
    logger.info(f"  Categorical features ({len(categorical_features)}): {categorical_features}")

    # STEP 2: Pre-process categorical data to ensure compatibility
    X_train_protected = X_train.copy()
    X_test_protected = X_test.copy()

    # Convert any problematic categorical data to strings and handle NaNs
    for cat_col in categorical_features:
        if cat_col in X_train_protected.columns:
            # Convert to string and handle NaNs
            X_train_protected[cat_col] = X_train_protected[cat_col].astype(str).replace('nan', 'missing')
            X_test_protected[cat_col] = X_test_protected[cat_col].astype(str).replace('nan', 'missing')

            # Check unique values
            unique_values = X_train_protected[cat_col].unique()
            logger.info(f"  '{cat_col}' unique values: {list(unique_values)}")

    logger.info("FIXING Initializing enhanced preprocessing pipeline...")

    # Initialize preprocessing pipeline
    preprocessor = PreprocessingPipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        experiment_dir=imputer_dir,
        imputation_method=imputation_method,
        feature_engineering_before_imputation=False,
        enable_feature_engineering = enable_feature_engineering
    )

    # STEP 4: Control feature engineering
    if not enable_feature_engineering:
        logger.info("DISABLING feature engineering completely")
        preprocessor.enable_feature_engineering = False

    # STEP 5: Fit preprocessing with enhanced error handling
    logger.info("ROTATE Fitting preprocessing pipeline with categorical preservation...")
    try:
        preprocessor.fit(X_train_protected, y_train)
        logger.info("SUCCESS Preprocessing pipeline fitted successfully")

        # Debug: Check what the preprocessor learned
        if hasattr(preprocessor, '_training_mode') and preprocessor._training_mode:
            logger.info(f"DATA Learned categorical modes: {preprocessor._training_mode}")

    except Exception as e:
        logger.error(f"ERROR Error fitting preprocessing pipeline: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")

        # Show detailed categorical info for debugging
        for col in categorical_features:
            logger.error(f"  '{col}': dtype={X_train_protected[col].dtype}, "
                         f"unique_count={X_train_protected[col].nunique()}, "
                         f"null_count={X_train_protected[col].isnull().sum()}")
        raise

    # STEP 6: Transform data with detailed logging
    logger.info("ROTATE Transforming data with categorical preservation...")
    try:
        # Transform the data
        X_train_processed = preprocessor.transform(X_train_protected)
        X_test_processed = preprocessor.transform(X_test_protected)
        y_train_processed = preprocessor.transform_target(y_train)
        y_test_processed = preprocessor.transform_target(y_test)

        logger.info("SUCCESS Data transformation completed successfully")
        logger.info(f"DATA Processed data shape: Train={X_train_processed.shape}, Test={X_test_processed.shape}")

    except Exception as e:
        logger.error(f"ERROR Error transforming data: {str(e)}")
        logger.error(f"Falling back to manual categorical handling...")

        # EMERGENCY FALLBACK: Manual categorical handling
        try:
            X_train_processed, X_test_processed = manual_categorical_handling(
                X_train_protected, X_test_protected, categorical_features, numeric_features
            )
            y_train_processed = y_train  # Use original target if transformer fails
            y_test_processed = y_test
            logger.info("SUCCESS Manual categorical handling completed")

        except Exception as fallback_error:
            logger.error(f"ERROR Manual fallback also failed: {str(fallback_error)}")
            raise

    # STEP 7: Final validation and cleanup
    logger.info("ANALYZING FINAL VALIDATION AND CLEANUP...")
    validate_processed_data(X_train_processed, X_test_processed, "AFTER FULL PREPROCESSING")

    # Check for any remaining non-numeric columns
    final_numeric_cols = X_train_processed.select_dtypes(include=[np.number]).columns
    final_non_numeric_cols = X_train_processed.select_dtypes(exclude=[np.number]).columns

    if len(final_non_numeric_cols) > 0:
        logger.error(f"ERROR STILL HAVE NON-NUMERIC COLUMNS: {list(final_non_numeric_cols)}")
        logger.info("FIXING Applying final emergency conversion...")

        # Final emergency conversion
        for col in final_non_numeric_cols:
            if col in X_train_processed.columns:
                try:
                    # Try to convert to numeric with error handling
                    X_train_processed[col] = pd.to_numeric(X_train_processed[col], errors='coerce')
                    X_test_processed[col] = pd.to_numeric(X_test_processed[col], errors='coerce')

                    # Fill any NaNs created by conversion with 0 (safer than median for unknown data)
                    nan_count = X_train_processed[col].isna().sum()
                    if nan_count > 0:
                        X_train_processed[col] = X_train_processed[col].fillna(0)
                        X_test_processed[col] = X_test_processed[col].fillna(0)
                        logger.info(f"SUCCESS Converted '{col}' to numeric, filled {nan_count} NaNs with 0")
                    else:
                        logger.info(f"SUCCESS Converted '{col}' to numeric")

                except Exception as conv_error:
                    logger.warning(f"WARNING Could not convert '{col}': {str(conv_error)}")
                    logger.warning(f"WARNING Dropping problematic column: '{col}'")
                    X_train_processed = X_train_processed.drop(columns=[col])
                    X_test_processed = X_test_processed.drop(columns=[col])

    # STEP 8: CRITICAL FIX - Extract only features that imputer was trained on
    logger.info("FIXING Ensuring feature consistency with imputer training...")

    if hasattr(preprocessor, '_removed_columns') and preprocessor._removed_columns:
        removed_columns = preprocessor._removed_columns
        logger.info(f"Imputer removed {len(removed_columns)} high-missing columns: {removed_columns}")

        # Filter processed data to match what imputer expects
        available_features = [f for f in X_train_processed.columns if f not in removed_columns]

        logger.info(f"Before filtering: {X_train_processed.shape[1]} features")
        X_train_processed = X_train_processed[available_features]
        X_test_processed = X_test_processed[available_features]
        logger.info(f"After filtering: {X_train_processed.shape[1]} features")

        # Verify no high-missing columns remain
        remaining_removed = [col for col in removed_columns if col in X_train_processed.columns]
        if remaining_removed:
            logger.error(f"ERROR: {len(remaining_removed)} removed columns still present: {remaining_removed}")
            X_train_processed = X_train_processed.drop(columns=remaining_removed)
            X_test_processed = X_test_processed.drop(columns=remaining_removed)
            logger.info(f"FIXED: Manually removed remaining columns. Final shape: {X_train_processed.shape}")
        else:
            logger.info("SUCCESS: No high-missing columns found in processed data")

    logger.info(f"Final data shape for feature selection: {X_train_processed.shape}")

    # STEP 9: Final data validation
    logger.info("FINAL DATA VALIDATION...")

    # Check for data leakage if possible
    if hasattr(preprocessor, 'validate_no_data_leakage'):
        try:
            leakage_check = preprocessor.validate_no_data_leakage(X_train_protected, X_test_protected)
            if leakage_check['has_leakage']:
                logger.error(f"ERROR DATA LEAKAGE DETECTED: {leakage_check['issues']}")
            else:
                logger.info("SUCCESS No data leakage detected")
        except Exception as e:
            logger.warning(f"Could not validate data leakage: {str(e)}")

    # Final NaN and infinite value checks
    train_nan_count = X_train_processed.isna().sum().sum()
    test_nan_count = X_test_processed.isna().sum().sum()

    if train_nan_count > 0 or test_nan_count > 0:
        logger.error(f"ERROR Found {train_nan_count} train + {test_nan_count} test NaN values after full preprocessing!")
        # Fill remaining NaNs with 0 as emergency measure
        X_train_processed = X_train_processed.fillna(0)
        X_test_processed = X_test_processed.fillna(0)
        logger.warning("FIXING Emergency: Filled remaining NaNs with 0")

    # Check for infinite values
    numeric_cols_final = X_train_processed.select_dtypes(include=[np.number]).columns
    if len(numeric_cols_final) > 0:
        train_inf_count = np.isinf(X_train_processed[numeric_cols_final]).sum().sum()
        test_inf_count = np.isinf(X_test_processed[numeric_cols_final]).sum().sum()

        if train_inf_count > 0 or test_inf_count > 0:
            logger.warning(f"FIXING Replacing {train_inf_count} train + {test_inf_count} test infinite values...")
            X_train_processed[numeric_cols_final] = X_train_processed[numeric_cols_final].replace([np.inf, -np.inf],
                                                                                                  [1e9, -1e9])
            X_test_processed[numeric_cols_final] = X_test_processed[numeric_cols_final].replace([np.inf, -np.inf],
                                                                                                [1e9, -1e9])

    logger.info(f"SUCCESS PREPROCESSING COMPLETE! Final shape: {X_train_processed.shape}")
    logger.info(f"SUCCESS All {len(X_train_processed.columns)} features are now numeric")


    # BOOTSTRAP EXPERIMENTS (if enabled)
    bootstrap_results = {}
    if enable_bootstrap:
        bootstrap_results = run_bootstrap_experiments(
            X_train_processed, X_test_processed,
            y_train_processed, y_test_processed,
            imputation_method, imputer_dir
        )
        save_bootstrap_metrics(bootstrap_results, imputation_method, experiment_dir)
    else:
        logger.info("Bootstrap experiments disabled")

    # Force ensure all data is numeric for bootstrap
    if enable_bootstrap:
        numeric_train_cols = X_train_processed.select_dtypes(include=[np.number]).columns
        non_numeric_train_cols = X_train_processed.select_dtypes(exclude=[np.number]).columns

        if len(non_numeric_train_cols) > 0:
            logger.error(
                f"ERROR CRITICAL ISSUE: Preprocessing failed to convert categorical columns: {list(non_numeric_train_cols)}")
            logger.error("This suggests a problem with your PreprocessingPipeline")

            # For now, force-fix by dropping non-numeric columns
            logger.warning("FIXING TEMPORARY FIX: Dropping non-numeric columns for the rest of the pipeline")
            X_train_processed = X_train_processed[numeric_train_cols]
            X_test_processed = X_test_processed[numeric_train_cols]

    # VALIDATION: Check for data leakage
    logger.info("Validating preprocessing for data leakage...")
    leakage_check = preprocessor.validate_no_data_leakage(X_train, X_test)
    if leakage_check['has_leakage']:
        logger.error(f"DATA LEAKAGE DETECTED: {leakage_check['issues']}")
        raise ValueError("Data leakage detected in preprocessing pipeline!")
    else:
        logger.info("SUCCESS No data leakage detected in preprocessing")

    # Check for NaN values after preprocessing
    train_nan_count = X_train_processed.isna().sum().sum()
    test_nan_count = X_test_processed.isna().sum().sum()

    if train_nan_count > 0 or test_nan_count > 0:
        logger.error(f"Found {train_nan_count} train + {test_nan_count} test NaN values after preprocessing!")
        raise ValueError("NaN values present after preprocessing - this indicates a pipeline error")

    logger.info("SUCCESS No NaN values found after preprocessing")

    # Check for infinite values only on numeric columns
    numeric_train_cols = X_train_processed.select_dtypes(include=[np.number]).columns
    numeric_test_cols = X_test_processed.select_dtypes(include=[np.number]).columns

    if len(numeric_train_cols) > 0:
        inf_count_train = np.isinf(X_train_processed[numeric_train_cols]).sum().sum()
        if inf_count_train > 0:
            logger.warning(f"Replacing {inf_count_train} infinite values in training data...")
            X_train_processed[numeric_train_cols] = X_train_processed[numeric_train_cols].replace([np.inf, -np.inf],
                                                                                                  [1e9, -1e9])

    if len(numeric_test_cols) > 0:
        inf_count_test = np.isinf(X_test_processed[numeric_test_cols]).sum().sum()
        if inf_count_test > 0:
            logger.warning(f"Replacing {inf_count_test} infinite values in test data...")
            X_test_processed[numeric_test_cols] = X_test_processed[numeric_test_cols].replace([np.inf, -np.inf],
                                                                                              [1e9, -1e9])

    # BOOTSTRAP EXPERIMENTS (Optional)
    bootstrap_results = {}
    if enable_bootstrap:
        bootstrap_results = run_bootstrap_experiments(
            X_train_processed, X_test_processed,
            y_train_processed, y_test_processed,
            imputation_method, imputer_dir
        )

        # Save bootstrap metrics
        save_bootstrap_metrics(bootstrap_results, imputation_method, experiment_dir)
    else:
        logger.info("Bootstrap experiments disabled")

    # STEP 8: Extract the features that the imputer actually used
    logger.info("FIXING Extracting features used by imputer...")

    # Get the features that the imputer was actually trained on
    if hasattr(preprocessor.imputer, '_filtered_feature_names'):
        # For FilteringIterativeImputer
        imputer_trained_features = preprocessor.imputer._filtered_feature_names
        removed_by_imputer = preprocessor.imputer.removed_columns
        logger.info(f"Imputer was trained on {len(imputer_trained_features)} features")
        logger.info(f"Imputer removed {len(removed_by_imputer)} high-missing features: {removed_by_imputer}")

        # Filter X_train_processed and X_test_processed to match imputer training
        available_features = [f for f in imputer_trained_features if f in X_train_processed.columns]

        if len(available_features) != len(imputer_trained_features):
            logger.warning(
                f"Feature mismatch: Expected {len(imputer_trained_features)}, found {len(available_features)}")
            logger.warning(f"Missing features: {set(imputer_trained_features) - set(available_features)}")

        # Use only the features that the imputer was trained on
        X_train_processed = X_train_processed[available_features]
        X_test_processed = X_test_processed[available_features]

        logger.info(
            f"SUCCESS Filtered processed data to {len(available_features)} features that imputer was trained on")

    elif hasattr(preprocessor, '_removed_columns') and preprocessor._removed_columns:
        # Fallback: manually remove the columns that were filtered out
        removed_columns = preprocessor._removed_columns
        remaining_features = [f for f in X_train_processed.columns if f not in removed_columns]

        X_train_processed = X_train_processed[remaining_features]
        X_test_processed = X_test_processed[remaining_features]

        logger.info(f"SUCCESS Manually filtered out {len(removed_columns)} removed features")
        logger.info(f"Using {len(remaining_features)} remaining features for feature selection")

    else:
        logger.info("No filtering information found - using all processed features")

    logger.info(f"Final processed data shape for feature selection: {X_train_processed.shape}")

    # FEATURE SELECTION LAYER - WITH LEAK PREVENTION
    logger.info("Starting Feature clustering on ALL engineered features...")
    logger.info("=" * 50)

    # Create feature selector for clustering
    cluster_selector = FeatureSelector(
        methods=['variance', 'mutual_info', 'random_forest'],
        random_state=42,
        experiment_dir=os.path.join(imputer_dir, 'clustering'),
        k_best=18
    )

    # Apply our FIXED clustering method to all engineered features
    clustered_features, feature_clusters, cluster_details = cluster_selector.cluster_and_select_features(
        X_train_processed, y_train_processed,
        correlation_threshold=0.70,
        selection_method='importance'
    )

    logger.info(f"SUCCESS Clustering selected {len(clustered_features)} diverse features from ALL engineered features")
    logger.info(f"Clustered features: {clustered_features}")

    # Apply clustering results to both train and test
    X_train_clustered = X_train_processed[clustered_features]
    X_test_clustered = X_test_processed[clustered_features]

    logger.info("Performing cross-validated feature selection...")
    logger.info("=" * 50)

    # Cross-validated feature selection to prevent data leakage
    from sklearn.model_selection import KFold
    feature_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    feature_vote_counts = {feature: 0 for feature in clustered_features}


    # CV to validate the clustered features
    for fold_idx, (fold_train_idx, fold_val_idx) in enumerate(feature_cv.split(X_train_clustered)):
        logger.info(f"CV refinement fold {fold_idx + 1}/5")

        X_fold_train = X_train_clustered.iloc[fold_train_idx]
        y_fold_train = y_train_processed.iloc[fold_train_idx]
        X_fold_val = X_train_clustered.iloc[fold_val_idx]
        y_fold_val = y_train_processed.iloc[fold_val_idx]

        # Quick feature importance on this fold
        from sklearn.ensemble import RandomForestRegressor
        rf_fold = RandomForestRegressor(n_estimators=50, random_state=42)
        rf_fold.fit(X_fold_train, y_fold_train)

        # Get feature importances and select top features for this fold
        importances = rf_fold.feature_importances_
        feature_importance_pairs = list(zip(X_fold_train.columns, importances))
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)

        # Vote for top 10-12 features in this fold
        top_features_this_fold = [feat for feat, imp in feature_importance_pairs[:12]]

        # Add votes
        for feature in top_features_this_fold:
            feature_vote_counts[feature] += 1

        logger.info(f"Fold {fold_idx + 1} voted for {len(top_features_this_fold)} features")

    # Select features that got votes from majority of folds
    min_votes = 3  # At least 3 out of 5 folds
    cv_selected_features = [feature for feature, votes in feature_vote_counts.items()
                            if votes >= min_votes]

    # If too few features, lower the threshold
    if len(cv_selected_features) < 8:
        min_votes = 2
        cv_selected_features = [feature for feature, votes in feature_vote_counts.items()
                                if votes >= min_votes]
    logger.info(f"SUCCESS CV refinement selected {len(cv_selected_features)} robust features")
    logger.info(f"CV selected features: {cv_selected_features}")

    # STEP 3: OPTIONAL RECURSIVE FEATURE ELIMINATION (on CV-selected features)
    logger.info("TARGET STEP 3: Optional recursive feature elimination...")
    logger.info("=" * 50)

    if len(cv_selected_features) > 10:  # Only if we have many features
        try:
            rfe_selector = FeatureSelector(
                methods=['variance', 'mutual_info', 'random_forest'],
                random_state=42,
                experiment_dir=os.path.join(imputer_dir, 'rfe'),
                k_best=10
            )

            X_train_cv_selected = X_train_clustered[cv_selected_features]

            rfe_features = rfe_selector.recursive_feature_selection_with_cv(
                X_train_cv_selected, y_train_processed,
                cv=3, scoring='r2'
            )

            final_selected_features = rfe_features
            logger.info(f"SUCCESS RFE refined to {len(final_selected_features)} final features")

        except Exception as e:
            logger.warning(f"RFE failed: {str(e)}, using CV-selected features")
            final_selected_features = cv_selected_features
    else:
        final_selected_features = cv_selected_features
        logger.info("SUCCESS Using CV-selected features (count already optimal)")

    logger.info(f"WINNER FINAL SELECTION: {len(final_selected_features)} features")
    logger.info(f"Final features: {final_selected_features}")

    # Log final feature analysis
    logger.info("DATA Final feature analysis:")
    original_count = sum(1 for f in final_selected_features if not any(
        f.startswith(prefix) for prefix in ['Log_', 'Log1p_', 'Sqrt_', 'Squared_', 'Ratio_']
    ))
    engineered_count = len(final_selected_features) - original_count
    ratio_count = sum(1 for f in final_selected_features if f.startswith('Ratio_'))

    logger.info(f"  Original features: {original_count}")
    logger.info(f"  Engineered features: {engineered_count}")
    logger.info(f"  Ratio features: {ratio_count}")


    # Option 2: Comprehensive feature analysis (includes multiple plots)
    try:
        logger.info("Creating comprehensive feature analysis...")
        analysis_results = create_advanced_feature_analysis(
            X=X_train_processed,
            y=y_train_processed,
            selected_features=final_selected_features,
            experiment_dir=imputer_dir
        )

        if analysis_results:
            logger.info("Feature analysis completed successfully")
            logger.info(f"High correlation pairs found: {len(analysis_results['high_correlation_pairs'])}")

            # Log the most important correlations with target
            target_corr_sorted = analysis_results['target_correlations'].abs().sort_values(ascending=False)
            logger.info("Top 5 features by correlation with target:")
            for i, (feature, corr) in enumerate(target_corr_sorted.head().items()):
                logger.info(f"  {i + 1}. {feature}: r = {analysis_results['target_correlations'][feature]:.3f}")

            # Log high correlation pairs
            if analysis_results['high_correlation_pairs']:
                logger.warning("Highly correlated feature pairs detected:")
                for pair in analysis_results['high_correlation_pairs']:
                    logger.warning(f"  {pair['feature1']} <-> {pair['feature2']}: r = {pair['correlation']:.3f}")

    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {str(e)}")
        logger.info("Falling back to simple correlation analysis...")

        # Fallback to simple version
        correlation_matrix, high_corr_pairs = create_simple_correlation_heatmap(
            X=X_train_processed,
            selected_features=final_selected_features,
            experiment_dir=imputer_dir,
            correlation_threshold=0.7
        )

    logger.info("Correlation analysis completed!")
    logger.info("=" * 50)

    # Prepare final data for modeling
    X_train_selected = X_train_processed[final_selected_features]
    X_test_selected = X_test_processed[final_selected_features]

    if enable_knn_augmentation:
        logger.info("=" * 60)
        logger.info("DATA AUGMENTATION FOR MODEL SELECTION")
        logger.info("=" * 60)

        from model_selection_augmentation import KNNDataAugmentor

        # Initialize augmentor with specified method
        augmentor = KNNDataAugmentor(
            n_neighbors=knn_n_neighbors,
            expansion_factor=knn_expansion_factor,
            noise_level=knn_noise_level,
            synthesis_method=knn_synthesis_method,  # NEW
            random_state=42
        )

        # Apply augmentation to training data only
        X_train_augmented, y_train_augmented = augmentor.fit_augment(
            X_train_selected,
            y_train_processed
        )

        # Get augmentation statistics
        aug_stats = augmentor.get_augmentation_stats(
            X_train_selected, X_train_augmented,
            y_train_processed, y_train_augmented
        )

        logger.info(f"SUCCESS Data Augmentation complete:")
        logger.info(f"  Method: {knn_synthesis_method.upper()}")
        logger.info(f"  Original samples: {aug_stats['original_samples']}")
        logger.info(f"  Synthetic samples: {aug_stats['synthetic_samples']}")
        logger.info(f"  Total samples: {aug_stats['augmented_samples']}")
        logger.info(f"  Expansion: {aug_stats['expansion_achieved']:.2f}x")
        logger.info(
            f"  Target mean shift: {aug_stats['target_stats']['mean_shift']:.4f} ({aug_stats['target_stats']['mean_shift_pct']:.2f}%)")

        # Save augmentation statistics
        augmentor.save_augmentation_report(aug_stats, imputer_dir, prefix=f'{imputation_method}_')

        # Use augmented data for model training
        X_train_for_modeling = X_train_augmented
        y_train_for_modeling = y_train_augmented

        # Store augmentation info
        augmentation_info = {
            'enabled': True,
            'method': knn_synthesis_method,
            'stats': aug_stats,
            'expansion_factor': knn_expansion_factor,
            'n_neighbors': knn_n_neighbors,
            'noise_level': knn_noise_level
        }

    else:
        logger.info("Data augmentation DISABLED - using original training data")
        X_train_for_modeling = X_train_selected
        y_train_for_modeling = y_train_processed

        augmentation_info = {
            'enabled': False,
            'method': None,
            'stats': None
        }

    logger.info("=" * 60)

    # MODEL LAYER WITH ENHANCED MULTI-CRITERIA EVALUATION
    logger.info("Training models with multi-criteria evaluation...")
    logger.info("=" * 50)

    # Configure model trainer
    model_trainer = ModelTrainer(
        experiment_dir=imputer_dir,
        random_state=42,
        n_jobs=-1,
    )

    # Train multiple models
    models = model_trainer.train_models(
        X_train_selected,
        y_train_processed,
        models=['random_forest', 'hist_gradient_boosting', 'xgboost', 'lightgbm', 'neural_network',  'svr', 'knn'],
    )

    # Create ensembles
    ensemble_model = model_trainer.create_ensemble(models, X_train_selected, y_train_processed)
    models['ensemble'] = ensemble_model

    enhanced_models = add_advanced_ensembles_to_pipeline(
        models, X_train_selected, y_train_processed, X_test_selected, y_test_processed
    )
    models.update(enhanced_models)

    # EVALUATION LAYER
    logger.info("Performing multi-criteria model evaluation...")
    logger.info("=" * 60)

    # Get cross-validator for CV metrics
    cross_validator = getattr(model_trainer, 'cross_validator', None)

    # Perform comprehensive evaluation
    weighted_results = integrate_weighted_selector_in_main(
        models=models,
        X_train_selected=X_train_selected,
        X_test_selected=X_test_selected,
        y_train_processed=y_train_processed,
        y_test_processed=y_test_processed,
        imputer_dir=imputer_dir,
        cross_validator=cross_validator
    )

    # Get the best model based on multi-criteria ranking
    best_model_name = weighted_results['best_model_name']

    logger.info(f"WINNER Best model selected by multi-criteria evaluation: {best_model_name}")

    # ===============================
    # FEATURE CORRELATION ANALYSIS
    # ===============================
    logger.info("Creating correlation heatmap for selected features...")
    logger.info("=" * 50)

    try:
        correlation_matrix = create_feature_correlation_heatmap(
            X=X_train_processed,
            selected_features=final_selected_features,
            experiment_dir=imputer_dir,
            figsize=(14, 12),
            save_name='selected_features_correlation_heatmap.png'
        )
        logger.info("Feature correlation heatmap created successfully")
    except Exception as e:
        logger.error(f"Error creating correlation heatmap: {str(e)}")

    # ===============================
    # FEATURE IMPORTANCE ANALYSIS
    # ===============================
    logger.info("Creating feature importance visualizations...")
    logger.info("=" * 50)

    try:
        # Initialize the feature importance visualizer
        feature_viz = FeatureImportanceVisualizer(experiment_dir=imputer_dir)

        logger.info(f"Analyzing feature importance for {len(final_selected_features)} selected features...")

        # Generate feature importance heatmap across all models
        feature_importance_df = feature_viz.generate_importance_heatmap(
            models=models,
            X=X_train_selected,
            y=y_train_processed,
            top_n=min(15, len(final_selected_features))
        )

        # Generate pairwise feature interaction analysis using best model
        if best_model_name in ['random_forest', 'xgboost', 'lightgbm', 'hist_gradient_boosting']:
            best_model = models[best_model_name]
            interaction_matrix, interaction_features = feature_viz.generate_pairwise_feature_interactions(
                model=best_model,
                X=X_train_selected,
                y=y_train_processed,
                top_n=min(10, len(final_selected_features))
            )
            logger.info("Feature interaction analysis completed")
        else:
            logger.info(f"Skipping interaction analysis for {best_model_name} (requires tree-based model)")

    except Exception as e:
        logger.error(f"Error in feature importance visualization: {str(e)}")

    logger.info("All visualizations completed!")
    logger.info("=" * 50)

    # TRADITIONAL EVALUATION FOR COMPARISON
    logger.info("Running traditional evaluation for comparison...")
    evaluator = ModelEvaluator(
        experiment_dir=imputer_dir,
        preprocessor=preprocessor,
        feature_selector=FeatureSelector(
            methods=['variance', 'mutual_info', 'random_forest'],
            random_state=42, experiment_dir=imputer_dir, k_best=15
        )
    )


    # Traditional evaluation
    traditional_results = evaluator.evaluate_models(
        models, X_train_selected, X_test_selected,
        y_train_processed, y_test_processed, use_cv=True
    )

    # Compare traditional vs multi-criteria selection
    traditional_best = evaluator.get_best_model_name(traditional_results)

    logger.info("=" * 60)
    logger.info("ANALYZING MODEL SELECTION COMPARISON:")
    logger.info(f"   Multi-Criteria Best: {best_model_name}")
    logger.info(f"   Traditional Best (R² only): {traditional_best}")

    if best_model_name != traditional_best:
        logger.info("   WARNING  Different models selected! Multi-criteria provides more balanced choice.")

        # Show why they're different
        mc_results = weighted_results[best_model_name]
        trad_results = weighted_results[traditional_best]

        logger.info(f"   Multi-Criteria Choice ({best_model_name}):")
        logger.info(
            f"      R²: {mc_results['test_r2']:.4f}, RMSE: {mc_results['test_rmse']:.4f}, Accuracy: {mc_results['prediction_accuracy']:.4f}")

        logger.info(f"   Traditional Choice ({traditional_best}):")
        logger.info(
            f"      R²: {trad_results['test_r2']:.4f}, RMSE: {trad_results['test_rmse']:.4f}, Accuracy: {trad_results['prediction_accuracy']:.4f}")
    else:
        logger.info("   SUCCESS Both methods agree on the best model!")

    logger.info("=" * 60)

    # Generate SHAP analysis for the multi-criteria best model
    if best_model_name in ['random_forest', 'hist_gradient_boosting', 'xgboost', 'lightgbm']:
        logger.info(f"Generating SHAP analysis for best model: {best_model_name}")
        evaluator.generate_shap_analysis(models[best_model_name], X_train_selected, best_model_name)

    # Create enhanced performance report
    evaluator.create_performance_report(weighted_results)

    # Save the best model (multi-criteria selection)
    best_model_path = os.path.join(imputer_dir, f"best_model_multi_criteria_{best_model_name}.pkl")
    joblib.dump(models[best_model_name], best_model_path)
    logger.info(f"Best model saved: {best_model_path}")

    # Save processed data
    joblib.dump(X_train_selected, os.path.join(experiment_dir, 'X_train.pkl'))
    joblib.dump(y_train_processed, os.path.join(experiment_dir, 'y_train.pkl'))
    joblib.dump(X_test_selected, os.path.join(experiment_dir, 'X_test.pkl'))
    joblib.dump(y_test_processed, os.path.join(experiment_dir, 'y_test.pkl'))

    logger.info(f"Processed training data saved to {experiment_dir}")

    # Return enhanced results
    return {
        'models': models,
        'evaluation_results': weighted_results,
        'traditional_results': traditional_results,
        'preprocessor': preprocessor,
        'X_train_selected': X_train_selected,
        'X_test_selected': X_test_selected,
        'y_train_processed': y_train_processed,
        'y_test_processed': y_test_processed,
        'best_model_name': best_model_name,
        'traditional_best_model': traditional_best,
        'selected_features': final_selected_features,
        'bootstrap_results': bootstrap_results,
        'feature_engineering_enabled': enable_feature_engineering,
        'clustered_features': clustered_features,
        'cv_selected_features': cv_selected_features,
        'feature_vote_counts': feature_vote_counts,
        'feature_mapping': {
            'original_features': list(X_train.columns),
            'processed_features': list(X_train_processed.columns) if 'X_train_processed' in locals() else [],
            'selected_features': final_selected_features,
            'final_training_features': list(X_train_selected.columns)
        }
    }



def main(enable_bootstrap=True,
         enable_feature_engineering=True,
         debug_categorical=True,
         enable_retraining=True,
         enable_knn_augmentation=False,  # NEW
         knn_expansion_factor=1.5,  # NEW
         knn_n_neighbors=5,  # NEW
         knn_noise_level=0.05,
         knn_synthesis_method='knn'
         ):
    """
    Main entry point for the ML pipeline with cross-validation.

    Parameters:
    -----------
    enable_bootstrap : bool, default=True
        Whether to run bootstrap experiments within the pipeline
    enable_feature_engineering : bool, default=True
        Whether to enable feature engineering in preprocessing
    """
    # Create experiment directory for outputs
    experiment_dir = create_experiment_dir()
    logger.info(f"Starting enhanced experiment with multi-criteria selection in {experiment_dir}")
    logger.info(f"Bootstrap experiments: {'ENABLED' if enable_bootstrap else 'DISABLED'}")
    logger.info(f"Feature engineering: {'ENABLED' if enable_feature_engineering else 'DISABLED'}")
    logger.info(f"Categorical debugging: {'ENABLED' if debug_categorical else 'DISABLED'}")
    logger.info(f"KNN augmentation: {'ENABLED' if enable_knn_augmentation else 'DISABLED'}")
    logger.info(f"Data augmentation: {'ENABLED' if enable_knn_augmentation else 'DISABLED'}")
    if enable_knn_augmentation:
        logger.info(f"  Method: {knn_synthesis_method}")
        logger.info(f"  Expansion factor: {knn_expansion_factor}x")
        logger.info(f"  Neighbors: {knn_n_neighbors}")
        logger.info(f"  Noise level: {knn_noise_level}")

    # DATA MANAGEMENT LAYER
    logger.info("Loading data...")
    data_loader = DataLoader()
    data = data_loader.load_excel(
        r"C:\Users\vjsin\OneDrive\Documents\Michigan Technological University\Research Related\Lead Bioavailability or BIoaccessibility\ML_Lead_Dataset.xlsx"
    )

    # ENHANCED DATA ANALYSIS
    if debug_categorical:
        logger.info("DATA ENHANCED DATA ANALYSIS:")
        logger.info(f"  Data shape: {data.shape}")
        logger.info(f"  Data types: {data.dtypes.value_counts()}")

        # Check each column type
        for col in data.columns:
            dtype = data[col].dtype
            unique_count = data[col].nunique()
            null_count = data[col].isnull().sum()

            if dtype == 'object':
                sample_values = data[col].dropna().unique()[:3]
                logger.info(
                    f"  INFO '{col}': {dtype}, {unique_count} unique, {null_count} nulls, samples: {list(sample_values)}")
            else:
                logger.info(f"  NUMERIC '{col}': {dtype}, {unique_count} unique, {null_count} nulls")

    # Data validation
    validator = DataValidator()
    validation_results = validator.validate(data)
    logger.info(f"Data validation results: {validation_results}")

    # Data versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data.to_csv(os.path.join(experiment_dir, f"data_snapshot_{timestamp}.csv"), index=False)

    # Extract features and target
    target_column = 'Bioaccessible Pb'
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Split data before preprocessing to prevent data leakage
    logger.info("Splitting data into train and test sets...")
    splitter = DataSplitter(random_state=42)
    X_train, X_test, y_train, y_test = splitter.train_test_split(X, y, test_size=0.2)

    logger.info(f"Original split: {len(X_train)} train, {len(X_test)} test samples")

    # RUN PIPELINES WITH BOTH IMPUTATION METHODS
    simple_results = run_pipeline_with_enhanced_model_selection(
        'simple', experiment_dir, X_train, X_test, y_train, y_test,
        enable_bootstrap=enable_bootstrap,
        enable_feature_engineering=enable_feature_engineering,
        enable_knn_augmentation=enable_knn_augmentation,  # NEW
        knn_expansion_factor=knn_expansion_factor,  # NEW
        knn_n_neighbors=knn_n_neighbors,  # NEW
        knn_noise_level=knn_noise_level,
        knn_synthesis_method=knn_synthesis_method  # NEW
    )

    iterative_results = run_pipeline_with_enhanced_model_selection(
        'iterative', experiment_dir, X_train, X_test, y_train, y_test,
        enable_bootstrap=enable_bootstrap,
        enable_feature_engineering=enable_feature_engineering,
        enable_knn_augmentation=enable_knn_augmentation,  # NEW
        knn_expansion_factor=knn_expansion_factor,  # NEW
        knn_n_neighbors=knn_n_neighbors,  # NEW
        knn_noise_level=knn_noise_level,
        knn_synthesis_method=knn_synthesis_method  # NEW
    )

    # COMPARE CROSS-VALIDATION RESULTS
    comparison_results = compare_weighted_results(simple_results, iterative_results, experiment_dir)

    # SAVE COMBINED BOOTSTRAP METRICS
    if enable_bootstrap:
        save_combined_bootstrap_metrics(simple_results, iterative_results, experiment_dir)

        # GENERATE PARTIAL DEPENDENCE PLOTS FOR WINNER
    winner_results = comparison_results['winner_results']
    winner_model = comparison_results['winner_model']

    logger.info(f"Generating partial dependence plots for winner: {winner_model}")
    generate_partial_dependence_for_best_cv_model(winner_results, winner_model, experiment_dir)

    # ENHANCED RETRAINING WITH WINNER
    if enable_retraining:
        logger.info("ROTATE Starting enhanced retraining with multi-criteria winner...")

        specified_indices = list(range(0, 30))
        available_for_random = list(set(range(len(data))) - set(specified_indices))
        random_indices = np.random.choice(available_for_random,size=20,replace=False).tolist()

        validation_indices = specified_indices + random_indices

        ## Just try one city
        #validation_indices = list(range(10,19))


        retrain_results_enhanced = retrain_with_explicit_control_enhanced(
            simple_results=simple_results,
            iterative_results=iterative_results,
            experiment_dir=experiment_dir,
            original_data=data,
            imputation_method=comparison_results['winner_method'].lower(),
            #imputation_method='iterative',
            model_name=winner_model,
            #model_name='bayesian_averaging',
            validation_strategy='indices',
            validation_config=validation_indices,
            test_size=0.15
        )

        retrain_results_stratified = integrate_stratified_validation_in_retrain_function(
            retrain_results=retrain_results_enhanced,
            model_name=winner_model,
            experiment_dir=experiment_dir
            )

        logger.info("SUCCESS Enhanced retraining completed!")

        # Run synthetic expansion
        retrain_results_enhanced = integrate_iterative_synthetic_validation_in_retrain_function(
            retrain_results=retrain_results_enhanced,
            model_name=winner_model,
            experiment_dir=experiment_dir,
            move_percentage=0.7,
            expansion_factor=5,  # 5x synthetic expansion
            synthesis_method='knn',
            min_iterations=200,  # At least 200 iterations
            min_sample_usage=30,  # Each sample used 10+ times
            random_state=42
        )


        logger.info("SUCCESS Synthetic expansion completed!")
        synthetic_r2 = retrain_results_enhanced['iterative_synthetic_expansion']['performance_comparison']['synthetic_val_r2']
        improvement = retrain_results_enhanced['iterative_synthetic_expansion']['performance_comparison']['improvement_over_original']
        logger.info(f"Synthetic expansion R²: {synthetic_r2:.4f}")
        logger.info(f"Total improvement: {improvement:+.4f}")


        retrain_results_enhanced_outlierhandling = integrate_outlier_removal_into_retrain_results(
            retrain_results=retrain_results_enhanced,
            model_name=winner_model,
            experiment_dir=experiment_dir,
            outlier_percentage=0.12,  # Remove worst 12% predict
            )

    logger.info("SUCCESS Complete outlier re-validation completed!")

    # Extract and log outlier removal results
    if 'complete_outlier_revalidation' in retrain_results_enhanced_outlierhandling:
        outlier_data = retrain_results_enhanced_outlierhandling['complete_outlier_revalidation']['results']
        comparison = outlier_data['comparison_metrics']

        original_r2 = comparison['original_r2']
        revalidation_r2 = comparison['revalidation_r2']
        outliers_removed = comparison['outliers_removed']
        improvement = comparison['improvement_revalidation_vs_original']

        logger.info(f"Outlier re-validation summary:")
        logger.info(f"  Original R² (all samples): {original_r2:.4f}")
        logger.info(f"  Re-validation R² (cleaned): {revalidation_r2:.4f}")
        logger.info(f"  Outliers removed: {outliers_removed}")
        logger.info(f"  Total improvement: {improvement:+.4f}")
    else:
        logger.warning("Outlier re-validation results not found in enhanced results")

    # FINAL REPORTING
    logger.info("=" * 60)
    logger.info("TARGET MULTI-CRITERIA MODEL SELECTION SUMMARY:")
    logger.info(f"   Winner: {comparison_results['winner_method']} Imputation")
    logger.info(f"   Best Model: {comparison_results['winner_model']}")
    logger.info(f"   Results saved to: {experiment_dir}")
    logger.info("=" * 60)

    logger.info(f"Enhanced experiment completed with multi-criteria selection.")

    return (experiment_dir, comparison_results,
            retrain_results_enhanced if enable_retraining else None,
            retrain_results_stratified if enable_retraining else None
            )


if __name__ == "__main__":
    # Control experiments here
    ENABLE_BOOTSTRAP = False  # Set to False to disable bootstrap experiments
    ENABLE_FEATURE_ENGINEERING = False  # Set to False to disable feature engineering
    DEBUG_CATEGORICAL = True
    ENABLE_RETRAINING = True

    # NEW: KNN Augmentation controls
    ENABLE_KNN_AUGMENTATION = True  # Set to True to enable
    KNN_SYNTHESIS_METHOD = 'hybrid'
    KNN_EXPANSION_FACTOR = 1.5  # 1.5 = 50% more samples
    KNN_N_NEIGHBORS = 7  # Number of neighbors
    KNN_NOISE_LEVEL = 0.03  # Noise level (0.05 = 5%)

    main(enable_bootstrap=ENABLE_BOOTSTRAP,
         enable_feature_engineering=ENABLE_FEATURE_ENGINEERING,
         debug_categorical=DEBUG_CATEGORICAL,
         enable_retraining=ENABLE_RETRAINING,
         enable_knn_augmentation=ENABLE_KNN_AUGMENTATION,
         knn_expansion_factor=KNN_EXPANSION_FACTOR,
         knn_n_neighbors=KNN_N_NEIGHBORS,
         knn_noise_level=KNN_NOISE_LEVEL,
         knn_synthesis_method=KNN_SYNTHESIS_METHOD
         )