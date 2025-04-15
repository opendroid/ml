from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import logging

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    total_rooms: str = 'total_rooms'
    total_bedrooms: str = 'total_bedrooms'
    population: str = 'population'
    households: str = 'households'


class ZeroToNaNTransformer(BaseEstimator, TransformerMixin):
    """Convert zero values to NaN for specified columns.

    This transformer is used for 'household' and 'rooms' columns to prevent
    division by zero errors in subsequent transformations. Zero values are
    converted to NaN and will be handled by the imputer in the pipeline.

    Attributes:
        None
    """

    def fit(self, X: npt.NDArray[np.float64],
            y: Optional[npt.NDArray[np.float64]] = None) -> (
                'ZeroToNaNTransformer'):
        """Fit the transformer (no-op).

        Args:
            X: Input features
            y: Target values (unused)

        Returns:
            self
        """
        return self

    def transform(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Convert zero values to NaN.

        Args:
            X: Input features

        Returns:
            Transformed features with zeros replaced by NaN
        """
        X = X.copy()
        zero_mask = X == 0
        if np.any(zero_mask):
            logger.warning(
                f"Found {np.sum(zero_mask)} zero values; converting to NaN")
        X[zero_mask] = np.nan
        return X


class PerHouseholdFeaturesAdder(BaseEstimator, TransformerMixin):
    """Add derived features based on household statistics.

    This transformer adds three new features:
    - rooms_per_household
    - population_per_household
    - bedrooms_per_room

    It handles edge cases like zero denominators by using median values
    computed during fitting.

    Attributes:
        feature_config: Configuration for feature names
        rooms_per_household_median_: Median rooms per household
        population_per_household_median_: Median population per household
        bedrooms_per_room_median_: Median bedrooms per room
    """

    def __init__(self, column_names: List[str],
                 feature_config: Optional[FeatureConfig] = None):
        """Initialize the transformer.

        Args:
            column_names: List of column names in the input data
            feature_config: Optional configuration for feature names
        """
        self.column_names = column_names
        self.feature_config = feature_config or FeatureConfig()

        # Get column indices
        self.rooms_ix = self.column_names.index(
            self.feature_config.total_rooms)
        self.bedrooms_ix = self.column_names.index(
            self.feature_config.total_bedrooms)
        self.population_ix = self.column_names.index(
            self.feature_config.population)
        self.household_ix = self.column_names.index(
            self.feature_config.households)

        # Initialize median attributes
        self.rooms_per_household_median_: float = 0.0
        self.population_per_household_median_: float = 0.0
        self.bedrooms_per_room_median_: float = 0.0

    def _compute_medians(self, X: npt.NDArray[np.float64]) -> None:
        """Compute median values for derived features.

        Args:
            X: Input features
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            rooms_per_household = X[:, self.rooms_ix] / X[:, self.household_ix]
            population_per_household = X[:, self.population_ix] / (
                X[:, self.household_ix])
            bedrooms_per_room = X[:, self.bedrooms_ix] / X[:, self.rooms_ix]

            self.rooms_per_household_median_ = np.nanmedian(
                rooms_per_household)
            self.population_per_household_median_ = np.nanmedian(
                population_per_household)
            self.bedrooms_per_room_median_ = np.nanmedian(bedrooms_per_room)

            self._log_median_warnings()

    def _log_median_warnings(self) -> None:
        """Log warnings for problematic median values."""
        for name, median in [
            ('rooms_per_household', self.rooms_per_household_median_),
            ('population_per_household',
             self.population_per_household_median_),
            ('bedrooms_per_room', self.bedrooms_per_room_median_)
        ]:
            if median == 0 or np.isnan(median):
                logger.warning(f"{name} median={median}; may cause issues")

    def fit(self, X: npt.NDArray[np.float64],
            y: Optional[npt.NDArray[np.float64]] = None) -> (
                'PerHouseholdFeaturesAdder'):
        """Fit the transformer by computing median values.

        Args:
            X: Input features
            y: Target values (unused)

        Returns:
            self
        """
        self._compute_medians(X)
        return self

    def transform(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Transform the input data by adding derived features.

        Args:
            X: Input features

        Returns:
            Transformed features with added derived features
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            # Compute derived features with zero handling
            rooms_per_household = self._compute_feature(
                X[:, self.rooms_ix],
                X[:, self.household_ix],
                self.rooms_per_household_median_,
                'rooms_per_household'
            )

            population_per_household = self._compute_feature(
                X[:, self.population_ix],
                X[:, self.household_ix],
                self.population_per_household_median_,
                'population_per_household'
            )

            bedrooms_per_room = self._compute_feature(
                X[:, self.bedrooms_ix],
                X[:, self.rooms_ix],
                self.bedrooms_per_room_median_,
                'bedrooms_per_room'
            )

        return np.c_[X,
                     rooms_per_household,
                     population_per_household,
                     bedrooms_per_room]

    def _compute_feature(
        self,
        numerator: npt.NDArray[np.float64],
        denominator: npt.NDArray[np.float64],
        median: float,
        feature_name: str
    ) -> npt.NDArray[np.float64]:
        """Compute a derived feature with zero handling.

        Args:
            numerator: Numerator values
            denominator: Denominator values
            median: Median value to use when denominator is zero
            feature_name: Name of the feature for logging

        Returns:
            Computed feature values
        """
        if np.any(denominator == 0):
            logger.warning(
                f"Found {np.sum(denominator == 0)} samples with zero "
                f" denominator for {feature_name}; using median {median}"
            )
        return np.where(denominator != 0, numerator / denominator, median)


def ca_housing_data_transformer(
    numerical_cols: List[str],
    categorical_col: List[str],
    categories: List[str],
    feature_config: Optional[FeatureConfig] = None
) -> ColumnTransformer:
    """Create a transformer for California housing data.

    The transformer consists of two pipelines:
    1. Numerical pipeline:
       - Convert zeros to NaN
       - Impute missing values with median
       - Add derived features
       - Scale features
    2. Categorical pipeline:
       - One-hot encode categorical features
       - Handle unknown categories

    Args:
        numerical_cols: List of numerical column names
        categorical_col: Name of categorical column
        categories: List of categories for each categorical feature
        feature_config: Optional configuration for feature names

    Returns:
        Configured ColumnTransformer
    """
    numerical_pipeline = Pipeline([
        ('zero_to_nan', ZeroToNaNTransformer()),
        ('imputer', SimpleImputer(strategy='median')),
        ('attr_adder', PerHouseholdFeaturesAdder(numerical_cols,
                                                 feature_config)),
        ('std_scaler', StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ('cat_encoder', OneHotEncoder(
            categories=[categories],
            sparse_output=False,
            handle_unknown='infrequent_if_exist'
        )),
    ])

    return ColumnTransformer([
        ('num_pipeline', numerical_pipeline, numerical_cols),
        ('cat_pipeline', categorical_pipeline, categorical_col),
    ])
