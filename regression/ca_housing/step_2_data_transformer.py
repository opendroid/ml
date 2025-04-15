from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import logging

# Configure logging
logger = logging.getLogger(__name__)


class ZeroToNaNTransformer(BaseEstimator, TransformerMixin):
    """Convert zero values to NaN for specified columns.
    Used for 'household' and 'rooms' columns.

    This is necessary for the PerHouseholdFeaturesAdder not to have
    division by zero errors. As it sets the zero and NaN values to the
    median of the training set, it will also apply to the test set.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if np.any(X == 0):
            logger.warning(
                f"Found {np.sum(X == 0)} zero values; converting to NaN")
        X[X == 0] = np.nan
        return X


class PerHouseholdFeaturesAdder(BaseEstimator, TransformerMixin):
    """Add three new attributes to the dataset:
    - rooms_per_household
    - population_per_household
    - bedrooms_per_room
    Handles potential zero medians and logs edge cases.
    """

    def __init__(self, column_names):
        self.column_names = column_names
        self.rooms_ix = self.column_names.index('total_rooms')
        self.bedrooms_ix = self.column_names.index('total_bedrooms')
        self.population_ix = self.column_names.index('population')
        self.household_ix = self.column_names.index('households')

    def fit(self, X, y=None):
        # Compute features for training data
        with np.errstate(divide='ignore', invalid='ignore'):
            rooms_per_household = X[:, self.rooms_ix] / X[:, self.household_ix]
            population_per_household = (X[:, self.population_ix] /
                                        X[:, self.household_ix])
            bedrooms_per_room = X[:, self.bedrooms_ix] / X[:, self.rooms_ix]

            # Compute medians, ignoring NaN
            self.rooms_per_household_median_ = np.nanmedian(
                rooms_per_household)
        self.population_per_household_median_ = np.nanmedian(
            population_per_household)
        self.bedrooms_per_room_median_ = np.nanmedian(bedrooms_per_room)
        # Warn if medians are zero or NaN
        if (self.rooms_per_household_median_ == 0 or
                np.isnan(self.rooms_per_household_median_)):
            logger.warning(
                f"rooms_per_household median="
                f"{self.rooms_per_household_median_}")
        if (self.population_per_household_median_ == 0 or
                np.isnan(self.population_per_household_median_)):
            logger.warning(
                f"population_per_household median="
                f"{self.population_per_household_median_}; may cause issues")
        if (self.bedrooms_per_room_median_ == 0 or
                np.isnan(self.bedrooms_per_room_median_)):
            logger.warning(
                f"bedrooms_per_room median={self.bedrooms_per_room_median_}")

        return self

    def transform(self, X):
        # Compute features, checking for zero denominators
        with np.errstate(divide='ignore', invalid='ignore'):
            # Log if households median caused zeros
            if np.any(X[:, self.household_ix] == 0):
                logger.warning(
                    f"Found {np.sum(X[:, self.household_ix] == 0)}"
                    "samples with households == 0 "
                    "for rooms_per_household; using median"
                    f"{self.rooms_per_household_median_}"
                )
            rooms_per_household = np.where(
                X[:, self.household_ix] != 0,
                X[:, self.rooms_ix] / X[:, self.household_ix],
                self.rooms_per_household_median_
            )

            # Reuse check for population_per_household
            if np.any(X[:, self.household_ix] == 0):
                logger.warning(
                    f"Found {np.sum(X[:, self.household_ix] == 0)}"
                    "samples with households == 0 "
                    "for population_per_household; using median"
                    f"{self.population_per_household_median_}"
                )
            population_per_household = np.where(
                X[:, self.household_ix] != 0,
                X[:, self.population_ix] / X[:, self.household_ix],
                self.population_per_household_median_
            )

            # Log if total_rooms median caused zeros
            if np.any(X[:, self.rooms_ix] == 0):
                logger.warning(
                    f"Found {np.sum(X[:, self.rooms_ix] == 0)}"
                    "samples with total_rooms == 0 "
                    "for bedrooms_per_room; using median"
                    f"{self.bedrooms_per_room_median_}"
                )
            bedrooms_per_room = np.where(
                X[:, self.rooms_ix] != 0,
                X[:, self.bedrooms_ix] / X[:, self.rooms_ix],
                self.bedrooms_per_room_median_
            )

        # Combine original features with new ones
        return np.c_[
            X,
            rooms_per_household,
            population_per_household,
            bedrooms_per_room
        ]


def ca_housing_data_transformer(numerical_cols, categorical_col, categories):
    """Create a transformer with the following steps:
    - Numerical pipeline: zero-to-NaN conversion, imputation, feature addition, scaling
    - Categorical pipeline: encoding with handling of unknown categories
    """
    numerical_pipeline = Pipeline([
        ('zero_to_nan', ZeroToNaNTransformer()),
        ('imputer', SimpleImputer(strategy='median')),
        ('attr_adder', PerHouseholdFeaturesAdder(numerical_cols)),
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
