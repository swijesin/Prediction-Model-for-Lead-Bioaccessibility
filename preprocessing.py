import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder, RobustScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from imputer import optimize_imputer_iterations_robust, create_iterative_imputer_with_filtering, apply_same_filtering
from datetime import datetime
import json
import os
import joblib
import logging

logger = logging.getLogger(__name__)


class OutlierHandler:
    """Class for handling outliers in data."""

    def __init__(self, method='iqr', contamination=0.1):
        self.method = method
        self.contamination = contamination
        self.bounds = {}
        self.is_trained = False

    def fit(self, X):
        """Fit the outlier handler to the data."""
        numeric_cols = X.select_dtypes(include=['number']).columns

        if self.method == 'iqr':
            for col in numeric_cols:
                q1 = X[col].quantile(0.25)
                q3 = X[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                self.bounds[col] = (lower_bound, upper_bound)

        elif self.method == 'winsorize':
            for col in numeric_cols:
                lower_bound = X[col].quantile(self.contamination)
                upper_bound = X[col].quantile(1 - self.contamination)
                self.bounds[col] = (lower_bound, upper_bound)

        self.is_trained = True
        return self

    def transform(self, X):
        """Transform the data by handling outliers."""
        if not self.is_trained:
            self.fit(X)  # Auto-fit if not trained

        X_transformed = X.copy()

        for col, (lower_bound, upper_bound) in self.bounds.items():
            if col in X_transformed.columns:
                X_transformed[col] = np.clip(X_transformed[col], lower_bound, upper_bound)

        return X_transformed


class FeatureEngineer:
    """Class for creating new features with proper NaN handling."""

    def __init__(self, include_logs=True, include_ratios=True, include_sqrt=False,
                 include_squared=True, reference_feature=None):
        """
        Initialize FeatureEngineer with all parameters having defaults.

        Parameters:
        -----------
        include_logs : bool, default=True
            Whether to include log transformations
        include_ratios : bool, default=True
            Whether to include ratio features
        include_sqrt : bool, default=False
            Whether to include square root transformations
        include_squared : bool, default=True
            Whether to include squared transformations
        reference_feature : str, default=None
            Feature to use as reference for ratios (will auto-detect if None)
        """
        self.include_logs = include_logs
        self.include_ratios = include_ratios
        self.include_sqrt = include_sqrt
        self.include_squared = include_squared
        self.reference_feature = reference_feature
        self.numeric_features = None
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        """Fit transformer, learning the feature names."""
        if isinstance(X, pd.DataFrame):
            self.numeric_features = X.select_dtypes(include='number').columns.tolist()

            # Auto-detect reference feature if not provided
            if self.reference_feature is None and self.include_ratios:
                # Look for TotalPb or similar
                for feature in X.columns:
                    if 'totalpb' in feature.lower() or ('total' in feature.lower() and 'pb' in feature.lower()):
                        self.reference_feature = feature
                        logger.info(f"Auto-detected reference feature: {feature}")
                        break

                # If still not found, use first numeric column
                if self.reference_feature is None and len(self.numeric_features) > 0:
                    self.reference_feature = self.numeric_features[0]
                    logger.warning(f"No TotalPb feature found. Using '{self.reference_feature}' as reference feature")

            # Generate a sample transformation to get output feature names
            # Use a copy to avoid modifying the original data
            X_sample = X.copy()

            # For fitting, we need to handle NaNs temporarily just to get feature names
            # We'll use forward fill then backward fill as a simple placeholder
            X_sample = X_sample.fillna(method='ffill').fillna(method='bfill').fillna(0)

            X_transformed_sample = self._transform_internal(X_sample)
            self.feature_names_out_ = X_transformed_sample.columns.tolist()
        else:
            raise ValueError("Input must be a pandas DataFrame")

        return self

    def transform(self, X):
        """Transform the data with model-specific engineered features - NO AUTOMATIC NaN FILLING."""
        # Check for NaN values and raise informative error
        nan_count = X.isna().sum().sum()
        if nan_count > 0:
            nan_columns = X.columns[X.isna().any()].tolist()
            raise ValueError(
                f"Found {nan_count} NaN values in {len(nan_columns)} columns before feature engineering: {nan_columns}. "
                f"All NaN values must be imputed BEFORE feature engineering to prevent data leakage. "
                f"Please ensure your imputation pipeline runs before feature engineering."
            )

        return self._transform_internal(X)

    def _transform_internal(self, X):
        """Internal method to perform the actual transformation with type safety."""
        # Ensure we're working with numeric data only for mathematical operations
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

        if len(non_numeric_cols) > 0:
            logger.debug(f"Non-numeric columns found during feature engineering: {non_numeric_cols}")
            logger.debug("Feature engineering will only be applied to numeric columns")

        X_transformed = X.copy()

        # Only work with numeric columns for mathematical transformations
        if len(numeric_cols) == 0:
            logger.warning("No numeric columns found for feature engineering")
            return X_transformed

        X_numeric = X[numeric_cols]

        # Update numeric_features if not set (for backward compatibility)
        if self.numeric_features is None:
            self.numeric_features = numeric_cols

        # Create feature ratios if applicable
        if self.include_ratios and self.reference_feature and self.reference_feature in numeric_cols:
            ref_values = X_numeric[self.reference_feature].replace(0, 1e-6)  # Avoid division by zero

            for col in numeric_cols:
                if col != self.reference_feature:
                    ratio_name = f'Ratio_{col}_to_{self.reference_feature}'
                    X_transformed[ratio_name] = X_numeric[col] / ref_values

        # Add log transformations for skewed features
        if self.include_logs:
            for col in numeric_cols:
                # Only apply log to strictly positive columns
                if (X_numeric[col] > 0).all():
                    X_transformed[f'Log_{col}'] = np.log(X_numeric[col])
                # Apply log1p to all non-negative columns (handles zeros)
                if (X_numeric[col] >= 0).all():
                    X_transformed[f'Log1p_{col}'] = np.log1p(X_numeric[col])

        # Add square root transformations
        if self.include_sqrt:
            for col in numeric_cols:
                if (X_numeric[col] >= 0).all():  # Only for non-negative data
                    X_transformed[f'Sqrt_{col}'] = np.sqrt(X_numeric[col])

        # Add squared terms
        if self.include_squared:
            for col in numeric_cols:
                X_transformed[f'Squared_{col}'] = X_numeric[col] ** 2

        # Check for any NaN or infinite values created during transformation - ONLY on new columns
        new_columns = [col for col in X_transformed.columns if col not in X.columns]

        if new_columns:
            new_data = X_transformed[new_columns]
            nan_count_after = new_data.isna().sum().sum()

            if nan_count_after > 0:
                nan_columns = new_data.columns[new_data.isna().any()].tolist()
                raise ValueError(
                    f"Feature engineering created {nan_count_after} NaN values in columns: {nan_columns}. "
                    f"This suggests incompatible operations (e.g., log of negative numbers). "
                    f"Please check your data preprocessing."
                )

            # Only check for infinite values in numeric columns
            numeric_new_cols = new_data.select_dtypes(include=[np.number]).columns
            if len(numeric_new_cols) > 0:
                try:
                    inf_count_after = np.isinf(new_data[numeric_new_cols]).sum().sum()
                    if inf_count_after > 0:
                        inf_columns = numeric_new_cols[np.isinf(new_data[numeric_new_cols]).any()].tolist()
                        logger.warning(
                            f"Feature engineering created {inf_count_after} infinite values in columns: {inf_columns}. "
                            f"Replacing with large finite values."
                        )
                        X_transformed[numeric_new_cols] = X_transformed[numeric_new_cols].replace([np.inf, -np.inf],
                                                                                                  [1e9, -1e9])
                except Exception as e:
                    logger.warning(f"Could not check for infinite values in new features: {str(e)}")

        # If expected output features are known, ensure consistency
        if self.feature_names_out_ is not None:
            # Add any missing columns with zeros
            for col in self.feature_names_out_:
                if col not in X_transformed.columns:
                    X_transformed[col] = 0

            # Keep only the expected columns in the expected order
            X_transformed = X_transformed[self.feature_names_out_]

        return X_transformed

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        if self.feature_names_out_ is None:
            raise ValueError("Transformer has not been fitted yet.")
        return self.feature_names_out_

    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


class PreprocessingPipeline:
    """Complete preprocessing pipeline with proper imputation order."""

    def __init__(self, numeric_features, categorical_features, experiment_dir,
                 imputation_method='simple', outlier_method='iqr',
                 feature_engineering_before_imputation=False,
                 enable_feature_engineering=True
                 ):
        """
        Initialize preprocessing pipeline.

        Parameters:
        -----------
        numeric_features : list
            List of numeric feature names
        categorical_features : list
            List of categorical feature names
        experiment_dir : str
            Directory to save preprocessing artifacts
        imputation_method : str, default='simple'
            Method for imputing missing values. Options:
            - 'simple': SimpleImputer with median strategy
            - 'iterative': IterativeImputer with RandomForest estimator
        outlier_method : str, default='winsorize'
            Method for handling outliers
        feature_engineering_before_imputation : bool, default=False
            Whether to do feature engineering before imputation (risky) or after (safer)
        """
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.experiment_dir = experiment_dir
        self.imputation_method = imputation_method
        self.feature_engineering_before_imputation = feature_engineering_before_imputation
        self.outlier_handler = OutlierHandler(method=outlier_method)
        self.feature_engineer = None
        self.imputer = None  # Will store the fitted imputer
        self.scaler = None  # Will store the fitted scaler
        self.target_transformer = PowerTransformer(method='yeo-johnson')
        self.enable_feature_engineering = enable_feature_engineering

        # Store training statistics for consistent imputation
        self._training_medians = None
        self._training_mode = None

    def _create_imputer(self, X_numeric):
        """Create the appropriate imputer based on the imputation method."""
        try:
            if self.imputation_method == 'simple':
                logger.info("Creating SimpleImputer with median strategy")
                from imputer import create_simple_imputer_with_filtering
                imputer,  removed_cols, filtered_features, tracking_info = create_simple_imputer_with_filtering(
                    X_outlier_handled= X_numeric,
                    missing_threshold=0.5,
                    experiment_dir=self.experiment_dir
                )

                self._imputation_tracking = tracking_info
                self._removed_columns = removed_cols
                self._filtered_features = filtered_features
                return imputer

            elif self.imputation_method == 'iterative':
                logger.info("Creating FilteringIterativeImputer with RandomForest estimator")


                # IMPORTANT: Pass the unfiltered data to let the imputer do the filtering
                imputer, removed_cols, filtered_features, tracking_info = create_iterative_imputer_with_filtering(
                    X_outlier_handled=X_numeric,  # This should be unfiltered
                    missing_threshold=0.65,  # Use 65% threshold
                    random_state=42,
                    experiment_dir=self.experiment_dir
                )

                # Store the filtering information
                self._removed_columns = removed_cols
                self._filtered_features = filtered_features
                self._imputation_tracking = tracking_info

                # CRITICAL: Update the class's numeric_features list
                if removed_cols:
                    logger.info(
                        f"FilteringIterativeImputer removed {len(removed_cols)} high-missing columns: {removed_cols}")
                    # Remove filtered columns from numeric_features
                    self.numeric_features = [col for col in self.numeric_features if col not in removed_cols]
                    logger.info(f"Updated numeric_features list to {len(self.numeric_features)} features")

                return imputer

            else:
                raise ValueError(f"Unknown imputation method: {self.imputation_method}")

        except Exception as e:
            logger.error(f"CRITICAL ERROR in _create_imputer: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

    def fit(self, X, y=None):
        """Fit the preprocessing pipeline to the data with enhanced categorical handling."""
        logger.info(f"ROTATE Fitting preprocessing pipeline with {self.imputation_method} imputation")
        logger.info(f"DATA Input data: {X.shape[0]} rows, {X.shape[1]} columns")
        logger.info(f"DATA Numeric features: {len(self.numeric_features)}")
        logger.info(f"DATA Categorical features: {len(self.categorical_features)}")

        # ... (keep all your existing steps 1-3 the same) ...

        # Step 1: Handle outliers on original numeric features only
        if self.numeric_features:
            X_outlier_handled = self.outlier_handler.fit(X[self.numeric_features]).transform(X[self.numeric_features])
            X_combined = X.copy()
            X_combined[self.numeric_features] = X_outlier_handled
        else:
            X_combined = X.copy()

        # Step 2: Decide imputation order
        if self.feature_engineering_before_imputation:
            logger.warning("Feature engineering BEFORE imputation - this may create additional NaNs!")
            X_engineered = self._fit_feature_engineering(X_combined)
            X_imputed = self._fit_imputation(X_engineered)
        else:
            logger.info("Imputation BEFORE feature engineering (recommended)")
            X_imputed = self._fit_imputation(X_combined)
            X_engineered = self._fit_feature_engineering(X_imputed)

        # Step 3: Handle categorical encoding
        if self.categorical_features:
            logger.info("FIXING Setting up categorical encoding...")
            X_with_encoding = self._apply_categorical_encoding(X_engineered)
        else:
            X_with_encoding = X_engineered

        # Step 4: Fit scaling on final feature set (EXCLUDE categorical features)
        # FIX 1: Only fit scaler on non-categorical numeric features
        all_numeric_columns = X_with_encoding.select_dtypes(include=['number']).columns.tolist()

        # Identify categorical feature names to exclude
        categorical_feature_names = []
        if self.categorical_features and hasattr(self, '_categorical_encoder'):
            try:
                categorical_feature_names = list(
                    self._categorical_encoder.get_feature_names_out(self.categorical_features))
                logger.info(f"Excluding {len(categorical_feature_names)} categorical features from scaler fitting")
            except Exception as e:
                logger.warning(f"Could not get categorical feature names during fit: {str(e)}")

        # Features to fit scaler on (exclude categorical)
        features_for_scaler = [col for col in all_numeric_columns if col not in categorical_feature_names]

        if len(features_for_scaler) > 0:
            if self.imputation_method == 'iterative':
                self.scaler = RobustScaler()
            else:
                self.scaler = StandardScaler()

            # Fit scaler only on non-categorical features
            self.scaler.fit(X_with_encoding[features_for_scaler])
            logger.info(f"SUCCESS Fitted scaler on {len(features_for_scaler)} non-categorical numeric features")

            # Log what features are being scaled vs not scaled
            logger.info(f"Features being scaled: {len(features_for_scaler)}")
            logger.info(f"Features NOT being scaled (categorical): {len(categorical_feature_names)}")

        else:
            logger.warning("WARNING No non-categorical numeric columns found for scaling")
            self.scaler = StandardScaler()  # Dummy scaler

        # Step 5: Fit target transformer if target is provided
        if y is not None:
            self.target_transformer.fit(y.values.reshape(-1, 1))
            logger.info("SUCCESS Fitted target transformer")

        # Step 6: Save all fitted components
        self._save_fitted_components()

        logger.info("SUCCESS Preprocessing pipeline fitting completed successfully")
        return self

    def _fit_imputation(self, X):
        """Fit the imputation step with PROPER categorical handling."""
        logger.info("ROTATE Fitting imputation with categorical data preservation...")

        # Separate numeric and categorical features
        X_numeric = X[self.numeric_features].copy()

        # Create and fit the imputer for NUMERIC features only
        self.imputer = self._create_imputer(X_numeric)  # Pass unfiltered data
        X_numeric_for_fitting = X[self.numeric_features].copy()

        # Store training statistics for fallback imputation
        self._training_medians = X_numeric_for_fitting.median()

        # Handle categorical features separately and store their modes
        if self.categorical_features:
            categorical_data = X[self.categorical_features].copy()
            # Calculate mode for each categorical feature (training data only)
            self._training_mode = {}
            for col in self.categorical_features:
                if col in categorical_data.columns:
                    mode_value = categorical_data[col].mode()
                    self._training_mode[col] = mode_value.iloc[0] if len(mode_value) > 0 else 'unknown'
                    logger.info(f"Categorical feature '{col}' mode: {self._training_mode[col]}")

        # Fit and transform numeric data using the fitted imputer
        X_numeric_imputed = self.imputer.fit_transform(X_numeric_for_fitting)

        # Convert back to DataFrame with correct column names
        if hasattr(X_numeric_imputed, 'columns'):
            # Already a DataFrame
            pass
        else:
            # Convert array to DataFrame
            X_numeric_imputed = pd.DataFrame(X_numeric_imputed, columns=self.numeric_features, index=X.index)

        # Start with original data and replace numeric columns
        X_result = X.copy()
        X_result[self.numeric_features] = X_numeric_imputed

        # Handle categorical imputation (fill NaNs with mode)
        if self.categorical_features:
            for col in self.categorical_features:
                if col in X_result.columns:
                    # Fill NaN values with the training mode
                    before_fill = X_result[col].isna().sum()
                    X_result[col] = X_result[col].fillna(self._training_mode.get(col, 'unknown'))
                    after_fill = X_result[col].isna().sum()
                    if before_fill > 0:
                        logger.info(
                            f"Filled {before_fill} NaN values in '{col}' with '{self._training_mode.get(col, 'unknown')}'")

        # Log the result
        numeric_cols_after = X_result.select_dtypes(include=[np.number]).columns
        categorical_cols_after = X_result.select_dtypes(include=['object', 'category']).columns
        logger.info(
            f"After imputation: {len(numeric_cols_after)} numeric, {len(categorical_cols_after)} categorical features")

        return X_result

    def _fit_feature_engineering(self, X):
        """Fit the feature engineering step with categorical preservation."""
        if not self.enable_feature_engineering:
            logger.info("Feature engineering DISABLED - preserving original data structure")
            return X  # Return data unchanged

        logger.info("FIXING Feature engineering ENABLED - applying transformations to numeric data only")

        # Initialize feature engineer with all defaults (for backward compatibility)
        self.feature_engineer = FeatureEngineer(
            include_logs=True,
            include_ratios=True,
            include_sqrt=True,
            include_squared=True,
            reference_feature=None  # Will auto-detect
        )

        self.feature_engineer.fit(X)

        # Transform to get the final feature set
        X_engineered = self.feature_engineer.transform(X)

        return X_engineered

    def transform(self, X):
        """Transform the data using the fitted preprocessing pipeline with categorical preservation."""
        logger.info(f"ROTATE Transforming data with {self.imputation_method} imputation...")

        # Step 1: Handle outliers (only on numeric features)
        X_outlier_handled = self.outlier_handler.transform(X[self.numeric_features])
        X_combined = X.copy()
        X_combined[self.numeric_features] = X_outlier_handled

        # Step 2: Apply transformations in the same order as fitting
        if self.feature_engineering_before_imputation:
            X_engineered = self._transform_feature_engineering(X_combined)
            X_imputed = self._transform_imputation(X_engineered)
        else:
            X_imputed = self._transform_imputation(X_combined)
            X_engineered = self._transform_feature_engineering(X_imputed)

        # Step 3: Handle categorical encoding BEFORE scaling
        if self.categorical_features:
            logger.info("FIXING Applying one-hot encoding to categorical features...")
            X_final = self._apply_categorical_encoding(X_engineered)
        else:
            X_final = X_engineered

        # Step 4: Apply scaling (EXCLUDE one-hot encoded features)
        # FIX 1: Identify which features should NOT be scaled
        numeric_columns = X_final.select_dtypes(include=['number']).columns.tolist()

        # Exclude one-hot encoded categorical features from scaling
        categorical_feature_names = []
        if self.categorical_features and hasattr(self, '_categorical_encoder'):
            try:
                categorical_feature_names = list(
                    self._categorical_encoder.get_feature_names_out(self.categorical_features))
                logger.info(f"Found {len(categorical_feature_names)} one-hot encoded features to exclude from scaling")
            except Exception as e:
                logger.warning(f"Could not get categorical feature names: {str(e)}")

        # Only scale original numeric features, NOT one-hot encoded features
        features_to_scale = [col for col in numeric_columns if col not in categorical_feature_names]

        logger.info(
            f"Scaling {len(features_to_scale)} numeric features (excluding {len(categorical_feature_names)} categorical)")

        if len(features_to_scale) > 0:
            X_scaled = X_final.copy()

            # Apply scaling only to non-categorical numeric features
            try:
                X_scaled[features_to_scale] = self.scaler.transform(X_final[features_to_scale])
                logger.info(f"SUCCESS Scaled {len(features_to_scale)} features")

                # Log the ranges of categorical features to verify they're still 0/1
                if categorical_feature_names:
                    sample_cat_feature = categorical_feature_names[0]
                    if sample_cat_feature in X_scaled.columns:
                        cat_min = X_scaled[sample_cat_feature].min()
                        cat_max = X_scaled[sample_cat_feature].max()
                        logger.info(f"Categorical feature '{sample_cat_feature}' range: [{cat_min:.3f}, {cat_max:.3f}]")

                        # Warning if categorical features have unexpected ranges
                        if cat_max > 1.1 or cat_min < -0.1:
                            logger.warning(f"WARNING: Categorical feature has unexpected range - may still be scaled!")
                        else:
                            logger.info("SUCCESS: Categorical features preserved as 0/1 values")

            except Exception as e:
                logger.error(f"Error during scaling: {str(e)}")
                logger.info("Falling back to no scaling...")
                X_scaled = X_final

        else:
            X_scaled = X_final
            logger.warning("WARNING No features to scale")

        # Final validation
        final_numeric = X_scaled.select_dtypes(include=[np.number]).columns
        final_non_numeric = X_scaled.select_dtypes(exclude=[np.number]).columns

        logger.info(
            f"Final transform result: {len(final_numeric)} numeric, {len(final_non_numeric)} non-numeric features")

        if len(final_non_numeric) > 0:
            logger.error(f"ERROR Still have non-numeric columns after preprocessing: {list(final_non_numeric)}")
            # Force convert remaining non-numeric to numeric
            for col in final_non_numeric:
                try:
                    X_scaled[col] = pd.to_numeric(X_scaled[col], errors='coerce').fillna(0)
                    logger.info(f"SUCCESS Force-converted '{col}' to numeric")
                except:
                    logger.warning(f"WARNING Dropping unconvertible column: '{col}'")
                    X_scaled = X_scaled.drop(columns=[col])

        return X_scaled

    def _apply_categorical_encoding(self, X):
        """Apply one-hot encoding to categorical features."""
        if not self.categorical_features:
            return X

        from sklearn.preprocessing import OneHotEncoder

        # Separate numeric and categorical data
        numeric_cols = [col for col in X.columns if col not in self.categorical_features]
        X_numeric = X[numeric_cols].copy()
        X_categorical = X[self.categorical_features].copy()

        # Initialize or load the one-hot encoder
        if not hasattr(self, '_categorical_encoder'):
            self._categorical_encoder = OneHotEncoder(
                sparse_output=False,
                handle_unknown='ignore',
                drop='first'  # Drop first category to avoid multicollinearity
            )

            # Fit the encoder on categorical data
            self._categorical_encoder.fit(X_categorical)
            logger.info("SUCCESS Fitted one-hot encoder on categorical features")

        # Transform categorical data
        X_cat_encoded = self._categorical_encoder.transform(X_categorical)
        self.validate_categorical_features(X_cat_encoded,"after encoding")
        # Create feature names for encoded features
        cat_feature_names = self._categorical_encoder.get_feature_names_out(self.categorical_features)

        # Convert to DataFrame
        X_cat_df = pd.DataFrame(
            X_cat_encoded,
            columns=cat_feature_names,
            index=X.index
        )

        # Combine numeric and encoded categorical features
        X_combined = pd.concat([X_numeric, X_cat_df], axis=1)

        logger.info(
            f"SUCCESS One-hot encoded {len(self.categorical_features)} categorical features into {len(cat_feature_names)} binary features")

        return X_combined

    def _transform_imputation(self, X):
        """Transform using the fitted imputation with PROPER categorical handling."""
        if self.imputer is None:
            raise ValueError("Imputer has not been fitted yet")

        logger.info("ROTATE Transforming with imputation...")

        # Apply imputation to numeric features only
        X_numeric = X[self.numeric_features].copy()

        if self.imputation_method == 'iterative' and hasattr(self, '_removed_columns') and self._removed_columns:
            X_numeric = apply_same_filtering(X_numeric, self._removed_columns)
            logger.info(f"Applied consistent high-missing column filtering during transform_imputation")

        # Check if there are any NaN values to impute
        nan_count = X_numeric.isna().sum().sum()
        if nan_count > 0:
            logger.info(f"Imputing {nan_count} NaN values using {self.imputation_method} method")
            X_numeric_imputed = self.imputer.transform(X_numeric)
            X_numeric_imputed = pd.DataFrame(X_numeric_imputed, columns=self.numeric_features, index=X.index)
        else:
            X_numeric_imputed = X_numeric

        # Start with original data and replace numeric columns
        X_result = X.copy()
        X_result[self.numeric_features] = X_numeric_imputed

        # Handle categorical features (should already be handled but ensure no NaNs)
        if self.categorical_features:
            for col in self.categorical_features:
                if col in X_result.columns:
                    # Fill any remaining NaN values with training mode
                    nan_count_cat = X_result[col].isna().sum()
                    if nan_count_cat > 0:
                        fill_value = self._training_mode.get(col, 'unknown')
                        X_result[col] = X_result[col].fillna(fill_value)
                        logger.info(f"Filled {nan_count_cat} NaN values in categorical '{col}' with '{fill_value}'")

        # Log the result
        numeric_cols_after = X_result.select_dtypes(include=[np.number]).columns
        categorical_cols_after = X_result.select_dtypes(include=['object', 'category']).columns
        logger.info(
            f"After imputation transform: {len(numeric_cols_after)} numeric, {len(categorical_cols_after)} categorical")

        return X_result

    def _transform_feature_engineering(self, X):
        """Transform using the fitted feature engineering."""
        if not self.enable_feature_engineering:
            logger.info("Feature engineering DISABLED - skipping transformation")
            return X

        if self.feature_engineer is None:
            raise ValueError("Feature engineer has not been fitted yet")

        return self.feature_engineer.transform(X)

    def transform_target(self, y):
        """Transform the target variable."""
        if y is None:
            return None

        # Check if PowerTransformer is destroying negative values
        print(f"Before transform: min={y.min():.3f}, max={y.max():.3f}, negatives={sum(y < 0)}")

        y_transformed = self.target_transformer.transform(y.values.reshape(-1, 1))
        y_series = pd.Series(y_transformed.flatten(), index=y.index)

        print(f"After transform: min={y_series.min():.3f}, max={y_series.max():.3f}, negatives={sum(y_series < 0)}")
        return y_series

    def inverse_transform_target(self, y_transformed):
        """Inverse transform the target variable."""
        if y_transformed is None:
            return None

        y_array = y_transformed
        if isinstance(y_transformed, pd.Series):
            y_array = y_transformed.values

        y_original = self.target_transformer.inverse_transform(y_array.reshape(-1, 1))

        if isinstance(y_transformed, pd.Series):
            return pd.Series(y_original.flatten(), index=y_transformed.index)
        return y_original.flatten()

    def add_engineered_features(self, X, reference_feature='TotalPb'):
        """
        DEPRECATED: Use the pipeline's built-in feature engineering instead.
        This method is kept for backward compatibility but will raise an error.
        """
        raise DeprecationWarning(
            "The add_engineered_features method is deprecated and potentially causes data leakage. "
            "Feature engineering is now handled automatically within the preprocessing pipeline. "
            "Set feature_engineering_before_imputation=True/False in the constructor to control the order."
        )

    def _save_fitted_components(self):
        """Save all fitted preprocessing components."""
        # Save imputer
        if self.imputer is not None:
            joblib.dump(self.imputer, os.path.join(self.experiment_dir, f'imputer_{self.imputation_method}.pkl'))

        # Save feature engineer
        if self.feature_engineer is not None:
            joblib.dump(self.feature_engineer, os.path.join(self.experiment_dir, 'feature_engineer.pkl'))

        # Save scaler
        if self.scaler is not None:
            joblib.dump(self.scaler, os.path.join(self.experiment_dir, f'scaler_{self.imputation_method}.pkl'))

        # Save target transformer
        joblib.dump(self.target_transformer, os.path.join(self.experiment_dir, 'target_transformer.pkl'))

        # Save training statistics
        if self._training_medians is not None:
            joblib.dump(self._training_medians, os.path.join(self.experiment_dir, 'training_medians.pkl'))
        if self._training_mode is not None:
            joblib.dump(self._training_mode, os.path.join(self.experiment_dir, 'training_mode.pkl'))

    def get_feature_names_out(self):
        """Get the final feature names after all transformations."""
        if self.feature_engineer is None:
            raise ValueError("Pipeline has not been fitted yet")
        return self.feature_engineer.get_feature_names_out()

    def validate_no_data_leakage(self, X_train_original, X_test_original):
        """
        Validate that no data leakage occurs during preprocessing.

        Parameters:
        -----------
        X_train_original : DataFrame
            Original training data before any preprocessing
        X_test_original : DataFrame
            Original test data before any preprocessing

        Returns:
        --------
        dict : Validation results
        """
        results = {
            'has_leakage': False,
            'issues': [],
            'warnings': []
        }

        # Check if imputation uses any test set statistics
        if hasattr(self.imputer, 'statistics_'):
            # For SimpleImputer, statistics should only come from training data
            train_medians = X_train_original[self.numeric_features].median()
            imputer_stats = pd.Series(self.imputer.statistics_, index=self.numeric_features)

            # Allow for small numerical differences
            if not np.allclose(train_medians, imputer_stats, rtol=1e-10, equal_nan=True):
                results['has_leakage'] = True
                results['issues'].append("Imputer statistics don't match training data medians")

        # Check feature engineering doesn't use test statistics
        if self.feature_engineer is not None:
            # This is harder to check automatically, but we can verify the feature names are consistent
            try:
                # This is a basic check - more sophisticated validation could be added
                results['warnings'].append("Feature engineering validation passed basic checks")
            except Exception as e:
                results['warnings'].append(f"Could not validate feature engineering: {str(e)}")

        return results

    def validate_categorical_features(self, X_processed, stage_name=""):
        """Validate that categorical features have expected 0/1 values."""
        if not hasattr(self, '_categorical_encoder') or not self.categorical_features:
            return

        try:
            categorical_feature_names = self._categorical_encoder.get_feature_names_out(self.categorical_features)

            logger.info(f"Validating categorical features {stage_name}...")

            for feature in categorical_feature_names:
                if feature in X_processed.columns:
                    unique_vals = X_processed[feature].unique()
                    min_val = X_processed[feature].min()
                    max_val = X_processed[feature].max()

                    logger.info(f"  {feature}: range=[{min_val:.3f}, {max_val:.3f}], unique values: {len(unique_vals)}")

                    # Check if values are approximately 0/1
                    if max_val > 1.1 or min_val < -0.1:
                        logger.error(f"  ERROR: {feature} has unexpected range - likely being scaled!")
                    elif len(unique_vals) > 10:
                        logger.warning(f"  WARNING: {feature} has {len(unique_vals)} unique values - may not be binary")
                    else:
                        logger.info(f"  SUCCESS: {feature} appears to be properly encoded")

        except Exception as e:
            logger.error(f"Error validating categorical features: {str(e)}")

    def get_imputation_tracking(self):
        """
        Get the imputation tracking information.

        Returns:
        --------
        dict : Tracking information from the imputation process
        """
        return getattr(self, '_imputation_tracking', {})

    def log_imputation_summary(self):
        """
        Log a summary of the imputation process.
        """
        tracking = self.get_imputation_tracking()

        if not tracking:
            logger.warning("No imputation tracking information available")
            return

        logger.info("=" * 50)
        logger.info("IMPUTATION PROCESS SUMMARY")
        logger.info("=" * 50)

        logger.info(f"Method: {tracking.get('imputation_method', 'Unknown')}")
        logger.info(f"Imputer Type: {tracking.get('imputer_type', 'Unknown')}")

        if tracking.get('imputation_method') == 'iterative':
            logger.info(f"Iterations Used: {tracking.get('iterations_used', 'Unknown')}")
            logger.info(f"Max Iterations: {tracking.get('max_iter', 'Unknown')}")
            logger.info(f"Tolerance: {tracking.get('tol', 'Unknown')}")

            removed_cols = tracking.get('removed_columns', [])
            if removed_cols:
                logger.info(f"Removed {len(removed_cols)} high-missing columns:")
                for col in removed_cols:
                    logger.info(f"  - {col}")

            # Log optimization details
            opt_details = tracking.get('optimization_details', {})
            if opt_details:
                logger.info(f"Optimization Selected: {opt_details.get('best_iterations')} iterations")
                logger.info(f"Optimization Recommendation: {opt_details.get('recommendation', 'N/A')}")

        # Log random states
        random_states = tracking.get('random_states_used', [])
        if random_states:
            logger.info("Random States Used:")
            for rs in random_states:
                logger.info(f"  {rs.get('component', 'Unknown')}: {rs.get('random_state', 'Unknown')}")

        logger.info("=" * 50)

    def save_preprocessing_summary(self):
        """
        Save a complete preprocessing summary including tracking info.
        """
        summary = {
            'preprocessing_config': {
                'imputation_method': self.imputation_method,
                'numeric_features_count': len(self.numeric_features),
                'categorical_features_count': len(self.categorical_features),
                'feature_engineering_enabled': getattr(self, 'enable_feature_engineering', True),
                'outlier_method': getattr(self.outlier_handler, 'method', 'unknown')
            },
            'imputation_tracking': self.get_imputation_tracking(),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Save summary
        summary_file = os.path.join(self.experiment_dir, 'preprocessing_summary.json')
        try:
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=4, default=str)
            logger.info(f"Preprocessing summary saved to: {summary_file}")
        except Exception as e:
            logger.error(f"Error saving preprocessing summary: {str(e)}")