import math
import time
from typing import List, Optional
from typing import Tuple, Dict, Any

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import statsmodels.api as sm
from IPython.display import display, Markdown
from matplotlib.lines import Line2D
from scipy import stats
from scipy.stats import pointbiserialr, shapiro
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import ParameterSampler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from tqdm.notebook import tqdm

CITRINE = "#e9cb0c"
NAPLES = "#ffd470"
CREAM = "#f3f6cb"
APPLE = "#9ea300"
MOSS = "#555610"
OLIVE = "#907E08"
HARVEST = "#E49F00"
PEAR = "#D1DC3A"
BACKGROUND_COLOR = "white"
ml_colors = [MOSS, APPLE, CREAM, NAPLES, CITRINE]
full_pallet = [APPLE, CITRINE, CREAM, HARVEST, OLIVE, NAPLES, MOSS, PEAR]
cmap = ListedColormap(ml_colors)


def check_df(
    dataframe: pd.DataFrame, head: int = 2, transpose: bool = True
) -> None:
    """
    Prints a comprehensive summary of a Pandas DataFrame, including shape,
    data types,
    a preview of the first and last few rows, null value counts, quantile
    statistics,
    and information on duplicate rows.

    Args:
            dataframe: The DataFrame to be analyzed.
            head: The number of rows to display from the beginning and end
            transpose: If True, transposes the quantile output forreadability
    """

    print("############## Shape ##############")
    display(dataframe.shape)

    print("\n############## Types ##############")
    types_df = pd.DataFrame({"Data Type": dataframe.dtypes})
    display(types_df)

    print("\n############## Quantiles ##############")
    quantiles = dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1])
    if transpose:
        display(quantiles.T)
    else:
        display(quantiles)

    print("\n############## Head ##############")
    display(dataframe.head(head))

    print("\n############## Tail ##############")
    display(dataframe.tail(head))

    print("\n############## Unique Value Check ##############")
    unique_value_df = pd.DataFrame(
        {
            "Column": dataframe.columns,
            "Unique Value Count": [
                dataframe[col].nunique() for col in dataframe.columns
            ],
        }
    )
    display(unique_value_df.sort_values("Unique Value Count", ascending=True))

    print("\n############## NA ##############")
    display(dataframe.isnull().sum().sort_values(ascending=True))

    print("\n############## Duplicate Rows ##############")
    duplicates_exist = dataframe.duplicated().any()
    if duplicates_exist:
        num_duplicates = dataframe.duplicated().sum()
        print(f"DataFrame contains {num_duplicates} duplicate rows.")

        duplicate_rows = dataframe[dataframe.duplicated(keep=False)]
        sorted_duplicate_rows = duplicate_rows.sort_values(
            by=list(dataframe.columns)
        )
        print("\nPreview of duplicate rows (sorted):")
        display(sorted_duplicate_rows.head())

    else:
        print("No duplicate rows found in the DataFrame.")


class NUMMiceImputer(BaseEstimator, TransformerMixin):
    """Imputes numerical data using MICE and optionally scales it.

    Attributes:
            numerical_features (List[str]): List of numerical feature names.
            columns_to_drop (Optional[List[str]]): List of columns to drop.
            max_iter (int): Maximum iterations for MICE.
            random_state (Optional[int]): Random state for MICE.
            estimator (Optional[object]): Estimator for MICE
            (e.g., RandomForestRegressor).
            no_inference (bool): Whether to skip scaling for inference
            (default False).
            verbose (int): Verbosity level (0 for no progress bar,
            >0 for progress bar).

    """

    def __init__(
        self,
        numerical_features: List[str],
        columns_to_drop: Optional[List[str]] = None,
        max_iter: int = 10,
        random_state: Optional[int] = None,
        estimator: Optional[object] = None,
        no_inference: bool = False,
        verbose: int = 0,
    ):
        """Initializes the MICENumerical transformer."""
        self.numerical_features = numerical_features
        self.columns_to_drop = columns_to_drop
        self.max_iter = max_iter
        self.random_state = random_state
        self.estimator = estimator
        self.no_inference = no_inference
        self.verbose = verbose
        self.scaler = StandardScaler()
        self.imputer = None

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "MICENumerical":
        """Fits the scaler and initializes the MICE imputer.

        Args:
                X (pd.DataFrame): Input DataFrame.
                y (Optional[pd.Series]): Target Series
                (not used in this transformer).

        Returns:
                MICENumerical: The fitted transformer.

        """
        self.scaler.fit(X[self.numerical_features])
        self.imputer = IterativeImputer(
            max_iter=self.max_iter,
            random_state=self.random_state,
            estimator=self.estimator,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms the data by imputing and optionally scaling.

        Args:
                X (pd.DataFrame): Input DataFrame.

        Returns:
                pd.DataFrame: Transformed DataFrame.

        """
        if self.columns_to_drop is not None:
            X = X.drop(columns=self.columns_to_drop)

        if self.verbose > 0:
            with tqdm(
                total=self.max_iter, desc="MICE Imputation", unit="iter"
            ) as pbar:
                for i in range(self.max_iter):
                    X[self.numerical_features] = self.imputer.fit_transform(
                        X[self.numerical_features]
                    )
                    pbar.update(1)
        else:
            X[self.numerical_features] = self.imputer.fit_transform(
                X[self.numerical_features]
            )

        if self.no_inference:
            X[self.numerical_features] = self.scaler.transform(
                X[self.numerical_features]
            )

        return X


class CATLogisticImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing values in categorical columns using predictive modeling
    (Logistic Regression).
    """

    def __init__(
        self,
        categorical_features: List[str],
        features_cols_dict: Dict[str, List[str]],
    ):
        """
        Initializes the PredictiveCategoricalImputer.

        Args:
                categorical_features (List[str]): List of categorical
                column names to impute.
                features_cols_dict (Dict[str, List[str]]): Dictionary mapping
                each categorical column
                        to a list of feature column names to use for prediction.
                        Keys of the dictionary should be column names from
                        `categorical_features`,
                        and values should be lists of column names from the
                        DataFrame to be used
                        as features to predict missing values in the
                        corresponding categorical column.
        """
        self.categorical_features = categorical_features
        self.features_cols_dict = features_cols_dict

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "PredictiveCategoricalImputer":
        """
        Fits the imputer (in this case, just returns self as models are trained
        in transform).

        Args:
                X (pd.DataFrame): Input DataFrame.
                y (Optional[pd.Series]): Target Series (not used in this
                unsupervised imputer).

        Returns:
                PredictiveCategoricalImputer: Returns self.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Imputes missing values in specified categorical columns using
        predictive modeling (Logistic Regression).

        Args:
                X (pd.DataFrame): Input DataFrame to transform.

        Returns:
                pd.DataFrame: DataFrame with imputed categorical columns.
        """
        X_imputed = X.copy()

        for target_col in tqdm(
            self.categorical_features, desc="Imputing categorical features"
        ):
            if target_col not in X_imputed.columns:
                continue
            if X_imputed[target_col].isnull().sum() == 0:
                continue

            features_cols = self.features_cols_dict.get(target_col)
            if not features_cols:
                continue

            train_data = X_imputed[X_imputed[target_col].notna()]
            predict_data = X_imputed[X_imputed[target_col].isna()]

            if predict_data.empty:
                continue

            X_train_original = train_data[features_cols]
            y_train = train_data[target_col]
            X_predict_original = predict_data[features_cols]

            X_train = pd.get_dummies(
                X_train_original, columns=features_cols, dummy_na=False
            )
            X_predict = pd.get_dummies(
                X_predict_original, columns=features_cols, dummy_na=False
            )

            train_cols = X_train.columns
            predict_cols = X_predict.columns
            missing_cols_predict = list(set(train_cols) - set(predict_cols))
            for c in missing_cols_predict:
                X_predict[c] = 0
            missing_cols_train = list(set(predict_cols) - set(train_cols))
            if missing_cols_train:
                print(
                    f"WARNING: Columns in predict_data not"
                    f" in train_data after "
                    f"one-hot encoding: {missing_cols_train}"
                )

            X_predict = X_predict[train_cols]

            model = LogisticRegression(
                solver="liblinear",
                multi_class="auto",
                random_state=42,
                max_iter=1000,
            )
            model.fit(X_train, y_train)
            predicted_values = model.predict(X_predict)

            X_imputed.loc[X_imputed[target_col].isna(), target_col] = (
                predicted_values
            )

        return X_imputed


class CATMiceImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing values in categorical columns using Multiple Imputation by
    Chained Equations (MICE) leveraging sklearn's IterativeImputer with
    either Logistic Regression or Gradient Boosting Classifier.
    """

    def __init__(
        self,
        categorical_features: List[str],
        features_cols_dict: Dict[str, List[str]],
        n_iterations: int = 10,
        max_iter_estimator: int = 100,
        estimator_type: str = "gb",
    ):
        """
        Initializes the CATMiceImputer.

        Args:
                categorical_features (List[str]): List of categorical
                        column names to impute.
                features_cols_dict (Dict[str, List[str]]): Dictionary mapping
                        each categorical column to a list of feature column
                        names to use for prediction in MICE.
                n_iterations (int, optional): Number of MICE iterations.
                        Defaults to 10.
                max_iter_estimator (int, optional): Maximum iterations
                 for the estimator.
                        For Logistic Regression, it's max_iter; for
                        Gradient Boosting, it's n_estimators.
                        Defaults to 100.
                estimator_type (str, optional): Type of estimator to use,
                 "lr" for Logistic Regression,
                  "gb" for Gradient Boosting. Defaults to "gb".
        """
        self.categorical_features = categorical_features
        self.features_cols_dict = features_cols_dict
        self.n_iterations = n_iterations
        self.max_iter_estimator = max_iter_estimator
        self.estimator_type = estimator_type
        self.imputer_models_ = {}

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "CATMiceImputer":
        """
        Fits the imputer (in this case, fitting is
        done in transform for IterativeImputer).
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Imputes missing categorical values using MICE with IterativeImputer.
        """
        X_imputed = X.copy()
        columns_to_impute = self.categorical_features

        for target_col in tqdm(
            columns_to_impute,
            desc=f"MICE Imputation " f"({self.estimator_type.upper()})",
        ):
            if target_col not in X_imputed.columns:
                continue
            if X_imputed[target_col].isnull().sum() == 0:
                continue

            features_cols = self.features_cols_dict.get(target_col)
            if not features_cols:
                continue

            data_for_imputation = X_imputed[
                features_cols + [target_col]
            ].copy()

            categorical_feature_cols_current_target = [
                col
                for col in features_cols + [target_col]
                if pd.api.types.is_categorical_dtype(data_for_imputation[col])
                or data_for_imputation[col].dtype == "object"
            ]
            data_encoded = pd.get_dummies(
                data_for_imputation,
                columns=categorical_feature_cols_current_target,
                dummy_na=False,
            )

            if self.estimator_type == "lr":
                estimator = LogisticRegression(
                    solver="liblinear",
                    multi_class="auto",
                    max_iter=self.max_iter_estimator,
                    random_state=42,
                )
            elif self.estimator_type == "gb":
                estimator = GradientBoostingClassifier(
                    n_estimators=self.max_iter_estimator,
                    random_state=42,
                    max_depth=3,
                )
            else:
                raise ValueError("estimator_type must be 'lr' or 'gb'")

            imputer = IterativeImputer(
                estimator=estimator,
                max_iter=self.n_iterations,
                random_state=42,
                verbose=0,
                initial_strategy="most_frequent",
                imputation_order="roman",
            )

            imputed_array = imputer.fit_transform(data_encoded)
            imputed_df = pd.DataFrame(
                imputed_array,
                columns=data_encoded.columns,
                index=data_encoded.index,
            )

            original_target_col_options = [
                col
                for col in imputed_df.columns
                if col.startswith(f"{target_col}_")
            ]
            if original_target_col_options:
                first_option_target_col = original_target_col_options[0]

                predicted_categories_indices = imputed_df[
                    original_target_col_options
                ].values.argmax(axis=1)
                predicted_categories = [
                    original_target_col_options[i].split(f"{target_col}_")[1]
                    for i in predicted_categories_indices
                ]

                X_imputed.loc[data_encoded.index, target_col] = (
                    predicted_categories
                )
            else:
                print(
                    f"Warning: No one-hot encoded columns found"
                    f" for target_col {target_col}. "
                    f"Imputation might not be as expected."
                )

            self.imputer_models_[target_col] = imputer

        return X_imputed


class CATBasicImputer(BaseEstimator, TransformerMixin):
    """Imputes missing values in categorical columns
    using the most frequent category."""

    def __init__(self, cat_cols: List[str]):
        """
        Args:
                cat_cols: List of categorical column names to impute.
        """
        self.cat_cols = cat_cols
        self.imputer = SimpleImputer(strategy="most_frequent")

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "CATBasicImputer":
        """Fits the imputer on the categorical columns of X."""
        self.imputer.fit(X[self.cat_cols])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Imputes missing values in categorical columns of X.

        Returns:
                DataFrame with imputed categorical columns.
        """
        X_transformed = X.copy()
        X_transformed[self.cat_cols] = X_transformed[self.cat_cols].replace(
            {None: np.nan}
        )
        X_transformed[self.cat_cols] = self.imputer.transform(
            X_transformed[self.cat_cols]
        )
        return X_transformed


class SplitColumn(BaseEstimator, TransformerMixin):
    """
    Transformer for splitting a column in a pandas DataFrame into multiple
    columns.

    This transformer takes a column containing delimited values and splits
    it into a specified number of new columns. It can optionally remove the
    original column after the split.

    Args:
            column_name (str): The name of the column to split.
            delimiter (str): The delimiter used to
            separate values in the column.
            num_splits (int): The number of new columns to create.
            new_column_names (List[str]): A list of names for the new columns.
            remove_original (bool, optional): Whether to remove the original
            column after splitting. Defaults to False.

    Example:
            >>> splitter = SplitColumn(
            ...     column_name="full_name",
            ...     delimiter="_",
            ...     num_splits=2,
            ...     new_column_names=["first_name", "last_name"],
            ...     remove_original=True
            ... )
            >>> df = pd.DataFrame({"full_name": ["John_Doe", "Jane_Doe"]})
            >>> splitter.transform(df)
               first_name last_name
            0        John       Doe
            1        Jane       Doe
    """

    def __init__(
        self,
        column_name: str,
        delimiter: str,
        num_splits: int,
        new_column_names: List[str],
        remove_original: bool = False,
    ):
        self.column_name = column_name
        self.delimiter = delimiter
        self.num_splits = num_splits
        self.new_column_names = new_column_names
        self.remove_original = remove_original

    def fit(self, X: pd.DataFrame, y=None) -> "SplitColumn":
        """
        "Fits" the transformer.

        This method doesn't perform any actual computation but is included
        for compatibility with scikit-learn's TransformerMixin interface.

        Args:
                X (pd.DataFrame): The input DataFrame.
                y (Ignored): Not used, present here for API consistency
                by convention.

        Returns:
                self: Returns the instance itself.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Performs the column splitting operation on the input DataFrame.

        Args:
                X (pd.DataFrame): The input DataFrame containing the column
                to be split.

        Returns:
                pd.DataFrame: The transformed DataFrame with the new columns
                added (and the original column optionally removed).

        Raises:
                ValueError: If the number of splits obtained is less than the
                specified `num_splits`.
        """
        X_copy = X.copy()
        split_columns = X_copy[self.column_name].str.split(
            self.delimiter, expand=True
        )

        if split_columns.shape[1] < self.num_splits:
            raise ValueError(
                f"Expected at least {self.num_splits}"
                f" splits, but got {split_columns.shape[1]}"
            )

        for i in range(self.num_splits):
            X_copy[self.new_column_names[i]] = split_columns[i]

        if self.remove_original:
            X_copy = X_copy.drop(self.column_name, axis=1)

        return X_copy


class BinColumn(BaseEstimator, TransformerMixin):
    """
    Bins a column in a pandas DataFrame into specified categories.

    Args:
            column_name (str): The name of the column to bin.
            bins (list): The bin edges (for "manual" strategy)
                                     or number of bins (for "quantile").
            labels (list): The labels for the bins.
            strategy (str, optional): The binning strategy to use.
            Options: "manual", "quantile".
            Defaults to "manual".
            remove_original (bool, optional): Whether to remove the original
            column after binning. Defaults to False.
            new_column_name (str, optional): The name of the new column create.
            If None, the name will be generated
            by appending "Group" to the original
            column name. Defaults to None.

    Example:
            >>> binner = BinColumn(
            ...     column_name="age",
            ...     bins=[0, 18, 30, 50, 100],
            ...     labels=["child", "young_adult", "adult", "senior"],
            ...     strategy="manual",
            ...     remove_original=True,
            ...     new_column_name="AgeCategory"
            ... )
            >>> df = pd.DataFrame({"age": [10, 25, 40, 60]})
            >>> binner.transform(df)
                             AgeCategory
            0          child
            1    young_adult
            2          adult
            3         senior

    """

    def __init__(
        self,
        column_name: str,
        bins: list,
        labels: list,
        strategy: str = "manual",
        remove_original: bool = False,
        new_column_name: str = None,
    ):
        self.column_name = column_name
        self.bins = bins
        self.labels = labels
        self.strategy = strategy
        self.remove_original = remove_original
        self.new_column_name = new_column_name

    def fit(self, X: pd.DataFrame, y=None) -> "BinColumn":
        """
        "Fits" the transformer.

        For "manual" and "quantile" strategies with pre-defined bins, this
        method doesn't do anything. For other strategies (e.g., equal-width),
        this would calculate the bin edges based on the data.

        Args:
                X (pd.DataFrame): The input DataFrame.
                y (Ignored): Not used, present here for API consistency by
                convention.

        Returns:
                self: Returns the instance itself.
        """
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the binning transformation to the specified column.

        Args:
                df (pd.DataFrame): The input DataFrame.

        Returns:
                pd.DataFrame: The modified DataFrame with the new binned column.
                        The original column is dropped if `remove_original`
                        is True.

        Raises:
                ValueError: If an invalid binning strategy is specified.
        """
        df_copy = df.copy()
        new_col_name = (
            (self.new_column_name)
            if (self.new_column_name)
            else (self.column_name + "Group")
        )

        if self.strategy == "manual":
            df_copy[new_col_name] = pd.cut(
                df_copy[self.column_name], bins=self.bins, labels=self.labels
            )
        elif self.strategy == "quantile":
            df_copy[new_col_name] = pd.qcut(
                df_copy[self.column_name],
                q=self.bins,
                labels=self.labels,
                duplicates="drop",
            )
        else:
            raise ValueError(f"Invalid binning strategy: {self.strategy}")

        if self.remove_original:
            df_copy = df_copy.drop(self.column_name, axis=1)
        return df_copy


class GroupSize(BaseEstimator, TransformerMixin):
    """
    Calculates the group size and identifies solo travelers.

    Adds two new columns to the DataFrame:
            - "GroupSize": The number of people in the passenger's group.
            - "TravellingSolo": A boolean column indicating whether
            the passenger is traveling alone.

    Requires the DataFrame to have "Group" and "PassengerId" columns,
     where "Group" likely represents
    a grouping identifier for passengers traveling together.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "GroupSize":
        """
        "Fits" the transformer. This method doesn't perform any
        actual computation
        but is required for compatibility with scikit-learn's TransformerMixin.

        Args:
                X (pd.DataFrame): The input DataFrame.
                y (Ignored): Not used, present here for API consistency by
                convention.

        Returns:
                self: Returns the instance itself.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates "GroupSize" and "TravellingSolo" and adds them
        to the DataFrame.

        Args:
                X (pd.DataFrame): The input DataFrame.

        Returns:
                pd.DataFrame: The modified DataFrame with the new "GroupSize"
                 and "TravellingSolo" columns.
        """
        X_copy = X.copy()
        X_copy["GroupSize"] = X_copy.groupby("Group")["PassengerId"].transform(
            "count"
        )
        X_copy["TravellingSolo"] = X_copy["GroupSize"] == 1
        return X_copy


class CalculateSpend(BaseEstimator, TransformerMixin):
    """
    Calculates the total spend and identifies passengers with zero spend.

    Adds two new columns to the DataFrame:
            - "TotalSpend": The sum of spending across "RoomService",
             "FoodCourt",
              "ShoppingMall", "Spa", and "VRDeck".
            - "ZeroSpend": A boolean column indicating whether "TotalSpend"
            is zero.
    """

    def __init__(
        self,
        total_spend_col: str = "TotalSpend",
        zero_spend_col: str = "ZeroSpend",
    ):
        self.total_spend_col = total_spend_col
        self.zero_spend_col = zero_spend_col

    def fit(self, X: pd.DataFrame, y=None) -> "CalculateSpend":
        """
        "Fits" the transformer (doesn't do anything in this case).
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates "TotalSpend" and "ZeroSpend" and adds them to the DataFrame.
        """
        X_copy = X.copy()
        X_copy[self.total_spend_col] = X_copy[
            ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
        ].sum(axis=1)
        X_copy[self.zero_spend_col] = X_copy[self.total_spend_col] == 0
        return X_copy


def impute_with_group_mode(df, group_col, target_cols):
    """
    Imputes missing values in the target columns with the mode of the same
    columns within each group defined by the group column.

    Args:
      df: The pandas DataFrame.
      group_col: The name of the column to group by.
      target_cols: A list of names of the columns to impute.

    Returns:
      The DataFrame with the imputed target columns.
    """
    for target_col in target_cols:
        df[target_col] = df.groupby(group_col)[target_col].transform(
            lambda x: x.mode()[0] if not x.mode().empty else None
        )
    return df


class TitanicNaNImputer(BaseEstimator, TransformerMixin):
    """
    Imputes NaN values in the Titanic dataset based on specific conditions.

    This transformer performs the following imputations:

    - Sets spending features (RoomService, FoodCourt, Spa, VRDeck, ShoppingMall)
      to 0 for passengers under 13 years old or in CryoSleep.
    - Imputes missing "LastName" and "HomePlanet" values using the mode within
      groups defined by the "Group" column (or another specified group column).
    - Sets "LastName" to "Unknown" for passengers travelling solo.
    - Fills NaN values in "FirstName" with "Unknown".

    Args:
            group_col (str): The name of the column to group by for mode
            imputation
                                              (default: "Group").

    Attributes:
            group_col (str): The name of the grouping column.
    """

    def __init__(self, group_col="Group"):
        """
        Initializes the TitanicNaNImputer with the specified grouping column.
        """
        self.group_col = group_col

    def fit(self, X, y=None):
        """
        This transformer does not require fitting, so it simply returns itself.
        """
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Performs the NaN imputations on the input DataFrame.

        Args:
                X (pd.DataFrame): The input DataFrame.
                y: Ignored. (Required by scikit-learn's TransformerMixin).

        Returns:
                pd.DataFrame: The DataFrame with imputed values.
        """
        df = X.copy()

        df["CryoSleep"] = df["CryoSleep"].astype(bool)
        spenditure_features = [
            "RoomService",
            "FoodCourt",
            "Spa",
            "VRDeck",
            "ShoppingMall",
        ]

        df.loc[(df["Age"] < 13), spenditure_features] = 0
        df.loc[(df["CryoSleep"] == True), spenditure_features] = 0

        df = impute_with_group_mode(
            df, group_col=self.group_col, target_cols=["HomePlanet"]
        )

        df.loc[
            ((df["TravellingSolo"] == True) & df["HomePlanet"].isna()),
            "HomePlanet",
        ] = "Unknown"

        df.loc[
            ((df["TravellingSolo"] == True) & df["LastName"].isna()),
            "LastName",
        ] = "Unknown"

        df.loc[(df["FirstName"].isna()), "FirstName"] = "Unknown"

        df = impute_with_group_mode(
            df, group_col=self.group_col, target_cols=["LastName"]
        )

        df = impute_with_group_mode(
            df, group_col=self.group_col, target_cols=["Side"]
        )

        df.loc[
            df["CabinDeck"].isin(["A", "B", "C", "T"])
            & df["HomePlanet"].isna(),
            "HomePlanet",
        ] = "Europa"

        df.loc[df["CabinDeck"] == "G", "HomePlanet"] = "Earth"

        return df


def display_repeated_values(
    df: pd.DataFrame, columns: List[str], head: int = 5, tail: int = 0
):
    """Displays repeated values for specified columns in a DataFrame."""

    for col in columns:
        value_counts = df[col].value_counts()
        repeated_values_df = value_counts[value_counts > 1].reset_index()
        print(f"Repeated values in '{col}':")
        if tail > 0:
            display(repeated_values_df.tail(tail))
        else:
            display(repeated_values_df.head(head))


def graph_all(
    df: pd.DataFrame,
    column_names: Optional[List[str]] = None,
    titles: Optional[List[str]] = None,
    max_cols: int = 3,
    figsize: Tuple[int, int] = (15, 12),
    plot_type: str = "histplot",
    bins: int = 30,
    hist_color: str = APPLE,
    line_color: str = MOSS,
) -> None:
    """
    Generates a grid of plots (histograms or boxplots) using Seaborn (sns)
    with custom styling, dynamically adjusting the number of rows and columns
    based on the number of features.

    Args:
            df: pandas DataFrame containing the data.
            column_names: A list of column names to plot.
                                      If None, all numeric columns are used for
                                      histograms,
                                      and all object/categorical columns are
                                      used for boxplots.
            titles: A list of titles for each subplot (optional).
                            If not provided, column names are used.
            max_cols: Maximum number of columns per row in
            the subplot grid (default: 3).
            figsize: Figure size (width, height) in inches (default: (15, 12)).
            plot_type: Type of plot: "histplot" (default) or "boxplot".
            bins: The number of bins in the histograms (default: 30).
            hist_color: The color of the histogram bars (default: APPLE).
            line_color: The color of the grid lines and outline (default: MOSS).

    Returns:
            None. Displays the plot.
    """

    if column_names is None:
        if plot_type == "histplot":
            column_names = df.select_dtypes(include=np.number).columns.tolist()
        elif plot_type == "boxplot":
            column_names = df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
        else:
            raise ValueError(
                "Invalid plot_type. Choose 'histplot' or" " 'boxplot'."
            )

    num_plots = len(column_names)
    if num_plots == 0:
        return

    num_rows = math.ceil(num_plots / max_cols)

    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=max_cols,
        figsize=figsize,
        facecolor=BACKGROUND_COLOR,
    )

    if num_plots == 1:
        axes = [axes]
    elif num_rows == 1:
        axes = axes.reshape(1, -1)[0]
    else:
        axes = axes.flatten()

    if titles is None:
        titles = [col.replace("_",
                              " ").title() for col in column_names]

    for i, (col, title) in enumerate(zip(column_names, titles)):
        ax = axes[i]
        ax.set_facecolor(BACKGROUND_COLOR)
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)

        data = df[col].dropna()

        if plot_type == "histplot":
            sns.histplot(
                data,
                bins=bins,
                ax=ax,
                color=hist_color,
                edgecolor=line_color,
                linewidth=1.5,
                kde=False,
                alpha=0.8,
            )
            ax.grid(
                which="major",
                axis="y",
                zorder=0,
                color=line_color,
                linestyle=":",
                dashes=(1, 5),
            )
            ax.set_xlabel(
                col.replace("_", " ").title(), fontsize=12, fontweight="bold"
            )
            ax.set_title(
                title, fontsize=14, fontweight="bold", fontfamily="serif"
            )
            ax.tick_params(axis="y", labelsize=10)

        elif plot_type == "boxplot":
            sns.boxplot(
                y=data,
                ax=ax,
                color=hist_color,
                width=0.6,
                **{
                    "boxprops": {"edgecolor": line_color, "linewidth": 1.5},
                    "medianprops": {"color": line_color, "linewidth": 1.5},
                    "whiskerprops": {"color": line_color, "linewidth": 1.5},
                    "capprops": {"color": line_color, "linewidth": 1.5},
                    "flierprops": {
                        "markerfacecolor": hist_color,
                        "markeredgecolor": line_color,
                        "markersize": 8,
                    },
                },
            )
            ax.set_ylabel(
                col.replace("_", " ").title(), fontsize=12, fontweight="bold"
            )
            ax.set_title(
                title, fontsize=14, fontweight="bold", fontfamily="serif"
            )
            ax.tick_params(axis="y", labelsize=10)

        else:
            raise ValueError(
                "Invalid plot_type." " Choose 'histplot' or 'boxplot'."
            )

    if num_plots < num_rows * max_cols:
        for i in range(num_plots, num_rows * max_cols):
            axes[i].set_axis_off()

    plt.tight_layout()
    plt.show()


def graph_by_group(
    df: pd.DataFrame,
    group_variable: str,
    target_variables: list,
    x_min_clip: float = None,
    x_max_clip: float = None,
    y_min_clip: float = None,
    y_max_clip: float = None,
    legend_x: float = 0.95,
    legend_y: float = 0.9,
) -> None:
    """
    Generates a grid of plots showing the distribution of each target variable
    based on the categories of a group variable.

    Args:
    df: pandas DataFrame containing the data.
    group_variable: The name of the grouping variable.
    target_variables: A list of names of the target variables to visualize.
    x_min_clip: The minimum value for the x-axis across all subplots.
    x_max_clip: The maximum value for the x-axis across all subplots.
    y_min_clip: The minimum value for the y-axis across all subplots.
    y_max_clip: The maximum value for the y-axis across all subplots.
    legend_x: The x-coordinate for the legend's position (default: 0.95).
    legend_y: The y-coordinate for the legend's position (default: 0.9).

    Returns:
    None. Displays the plot.
    """

    num_targets = len(target_variables)
    num_cols = 1
    num_rows = (num_targets + 1) // num_cols

    fig = plt.figure(
        figsize=(12, 5 * num_rows), dpi=150, facecolor=BACKGROUND_COLOR
    )
    gs = fig.add_gridspec(num_rows, num_cols)
    gs.update(wspace=0.1, hspace=0.4)

    categories = df[group_variable].unique()

    predefined_colors = [MOSS, CITRINE, APPLE, CREAM, NAPLES]

    if len(categories) <= 5:
        colors_list = predefined_colors
    else:
        raise ValueError(
            f"Not enough colors defined in the palette for variable"
            f" '{group_variable}'. "
            f"Please add at least {len(categories) - len(predefined_colors)}"
            f" more colors to the palette. "
            f"Currently, there are {len(categories)} unique categories "
            f"in '{group_variable}', "
            f"but only {len(predefined_colors)} colors are defined."
        )

    colors = {
        category: colors_list[i % len(colors_list)]
        for i, category in enumerate(categories)
    }

    if x_min_clip is None or x_max_clip is None:
        x_min_clip_calc = (
            df[target_variables].min().min()
            if all(
                pd.api.types.is_numeric_dtype(df[var])
                for var in target_variables
            )
            else None
        )
        x_max_clip_calc = (
            df[target_variables].max().max()
            if all(
                pd.api.types.is_numeric_dtype(df[var])
                for var in target_variables
            )
            else None
        )

    if y_min_clip is None or y_max_clip is None:
        y_min_clip_calc = 0
        y_max_clip_calc = 0

        for target_variable in target_variables:
            if not pd.api.types.is_numeric_dtype(df[target_variable]):
                for category in categories:
                    counts = df[df[group_variable] == category][
                        target_variable
                    ].value_counts()
                    y_max_clip_calc = max(y_max_clip_calc, counts.max())

        y_max_clip_calc *= 1.1

    x_min_clip = x_min_clip_calc if x_min_clip is None else x_min_clip
    x_max_clip = x_max_clip_calc if x_max_clip is None else x_max_clip
    y_min_clip = y_min_clip_calc if y_min_clip is None else y_min_clip
    y_max_clip = y_max_clip_calc if y_max_clip is None else y_max_clip

    for i, target_variable in enumerate(target_variables):
        row = i // num_cols
        col = i % num_cols
        ax = fig.add_subplot(gs[row, col])

        ax.set_facecolor(BACKGROUND_COLOR)
        ax.tick_params(axis="y", left=False)
        ax.get_yaxis().set_visible(False)
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)

        legend_elements = []

        for category in categories:
            subset = df[df[group_variable] == category]

            if pd.api.types.is_numeric_dtype(df[target_variable]):
                sns.kdeplot(
                    subset[target_variable],
                    ax=ax,
                    color=colors[category],
                    fill=True,
                    linewidth=2,
                    ec=MOSS,
                    alpha=0.7,
                    zorder=0,
                )
                ax.axvline(
                    subset[target_variable].median(),
                    color=colors[category],
                    linestyle="--",
                    zorder=4,
                )

                ax.set_xlim(x_min_clip, x_max_clip)
            else:
                counts = subset[target_variable].value_counts()
                total_count = counts.sum()
                percentages = (counts / total_count) * 100
                bars = ax.bar(
                    counts.index.astype(str),
                    counts.values,
                    color=colors[category],
                    edgecolor=MOSS,
                    linewidth=1,
                )
                for bar, percentage in zip(bars, percentages):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height,
                        f"{percentage:.1f}%",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

                ax.set_ylim(y_min_clip, y_max_clip)

        ax.grid(
            which="major",
            axis="x",
            zorder=0,
            color=MOSS,
            linestyle=":",
            dashes=(1, 5),
        )
        ax.set_title(
            f"'{target_variable}' Distribution by"
            f" '{group_variable}' feature",
            fontsize=14,
            fontweight="bold",
            fontfamily="serif",
        )

        for category in categories:
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    color=colors[category],
                    lw=2,
                    label=f"{group_variable} = {category}",
                )
            )
        fig.legend(
            handles=legend_elements,
            loc="upper right",
            bbox_to_anchor=(legend_x, legend_y),
            fontsize=10,
        )

    plt.tight_layout()
    plt.show()


def scatterplot(
    df: pd.DataFrame,
    x_variable: str,
    y_variable: str,
    x_min_clip: float = None,
    x_max_clip: float = None,
    y_min_clip: float = None,
    y_max_clip: float = None,
    padding: float = 0.05,
) -> None:
    """
    Generates a scatterplot of two variables.

    Args:
      df: pandas DataFrame containing the data.
      x_variable: The name of the variable for the x-axis.
      y_variable: The name of the variable for the y-axis.
      x_min_clip: The minimum value for the x-axis.
      x_max_clip: The maximum value for the x-axis.
      y_min_clip: The minimum value for the y-axis.
      y_max_clip: The maximum value for the y-axis.
      padding: The proportion of padding to add to the axes limits.

    Returns:
      None. Displays the plot.
    """

    fig, ax = plt.subplots(
        figsize=(12, 8), dpi=150, facecolor=BACKGROUND_COLOR
    )

    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plt.scatter(
        df[x_variable], df[y_variable], color=CITRINE, alpha=0.7, zorder=2
    )

    if x_min_clip is None:
        x_min_clip = df[x_variable].min()
        x_range = df[x_variable].max() - x_min_clip
        x_min_clip -= x_range * padding
    if x_max_clip is None:
        x_max_clip = df[x_variable].max()
        x_range = x_max_clip - df[x_variable].min()
        x_max_clip += x_range * padding
    if y_min_clip is None:
        y_min_clip = df[y_variable].min()
        y_range = df[y_variable].max() - y_min_clip
        y_min_clip -= y_range * padding
    if y_max_clip is None:
        y_max_clip = df[y_variable].max()
        y_range = y_max_clip - df[y_variable].min()
        y_max_clip += y_range * padding

    plt.xlim(x_min_clip, x_max_clip)
    plt.ylim(y_min_clip, y_max_clip)

    ax.set_facecolor(BACKGROUND_COLOR)
    ax.grid(False)
    ax.grid(
        which="major",
        axis="both",
        zorder=0,
        color=MOSS,
        linestyle=":",
        dashes=(1, 5),
    )
    ax.set_xlabel(
        x_variable, fontsize=14, fontweight="bold", fontfamily="serif"
    )
    ax.set_ylabel(
        y_variable, fontsize=14, fontweight="bold", fontfamily="serif"
    )
    ax.set_title(
        f"Scatterplot of '{y_variable}' vs. '{x_variable}'",
        fontsize=16,
        fontweight="bold",
        fontfamily="serif",
    )
    fig.tight_layout()
    plt.show()


def stacked_barcharts_by_group(
    df: pd.DataFrame,
    group_variable: str,
    target_variables: list,
    transpose: bool = False,
    legend_x: float = 0.95,
    legend_y: float = 0.95,
) -> None:
    """
    Generates a grid of *normalized* stacked bar charts.
    """

    num_targets = len(target_variables)
    num_cols = 2
    num_rows = (num_targets + 1) // num_cols

    fig = plt.figure(
        figsize=(12, 6 * num_rows), dpi=150, facecolor=BACKGROUND_COLOR
    )
    gs = fig.add_gridspec(num_rows, num_cols)
    gs.update(wspace=0.3, hspace=0.3)
    legend_data = None

    for i, target_variable in enumerate(target_variables):
        row = i // num_cols
        col = i % num_cols
        ax = fig.add_subplot(gs[row, col])

        ax.set_facecolor(BACKGROUND_COLOR)
        ax.tick_params(axis="y", left=False)
        ax.get_yaxis().set_visible(False)
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)

        df_dropped = df.dropna(subset=[target_variable])

        if not transpose:
            grouped_data = (
                df_dropped.groupby([group_variable, target_variable])
                .size()
                .unstack(fill_value=0)
            )
            group_totals = grouped_data.sum(axis=1)
            percentages = grouped_data.div(group_totals, axis=0) * 100
            categories = df_dropped[target_variable].unique()
            num_categories = len(categories)

            if num_categories <= 5:
                colors = [APPLE, CITRINE, MOSS, CREAM, NAPLES]
            else:
                raise ValueError(
                    "Not enough colors defined in the palette. "
                    "Please add more colors to the `colors` list "
                    "to accommodate all categories in your data. "
                    f"The variable '{target_variable}' "
                    f"has {num_categories} unique categories, "
                    f"but only {len(colors)} colors are defined."
                )

            bottom = np.zeros(len(grouped_data))
            for j, category in enumerate(categories):
                values = percentages[category].values
                bars = ax.bar(
                    grouped_data.index.astype(str),
                    values,
                    bottom=bottom,
                    label=category,
                    color=colors[j],
                    edgecolor=MOSS,
                    linewidth=2,
                )

                for bar, percentage in zip(bars, values):
                    height = bar.get_height()
                    if height > 0:
                        label_position = bar.get_y() + height / 2
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            label_position,
                            f"{percentage:.1f}%",
                            ha="center",
                            va="center",
                            fontsize=9,
                            color="black",
                        )

                bottom += values

            ax.set_ylim(0, 100)

            ax.set_title(
                f"'{target_variable}' Distribution by "
                f"'{group_variable}'",
                fontsize=14,
                fontweight="bold",
                fontfamily="serif",
                y=1.05,
            )

        else:
            grouped_data = (
                df_dropped.groupby([target_variable, group_variable])
                .size()
                .unstack(fill_value=0)
            )
            group_totals = grouped_data.sum(axis=1)
            percentages = grouped_data.div(group_totals, axis=0) * 100
            categories = df_dropped[group_variable].unique()
            num_categories = len(categories)

            if num_categories <= 5:
                colors = [APPLE, CITRINE, MOSS, CREAM, NAPLES]
            else:
                raise ValueError(
                    "Not enough colors defined in the palette. "
                    "Please add more colors to the `colors` list "
                    "to accommodate all categories in your data. "
                    f"The variable '{target_variable}' "
                    f"has {num_categories} unique categories, "
                    f"but only {len(colors)} colors are defined."
                )

            bottom = np.zeros(len(grouped_data))
            for j, category in enumerate(categories):
                values = percentages[category].values
                bars = ax.bar(
                    grouped_data.index.astype(str),
                    values,
                    bottom=bottom,
                    label=category,
                    color=colors[j],
                    edgecolor=MOSS,
                    linewidth=2,
                )

                for bar, percentage in zip(bars, values):
                    height = bar.get_height()
                    if height > 0:
                        label_position = bar.get_y() + height / 2
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            label_position,
                            f"{percentage:.1f}%",
                            ha="center",
                            va="center",
                            fontsize=9,
                            color="black",
                        )

                bottom += values

            ax.set_ylim(0, 100)

            ax.set_title(
                f"'{group_variable}' Distribution by" 
                f" '{target_variable}'",
                fontsize=14,
                fontweight="bold",
                fontfamily="serif",
                y=1.05,
            )

        ax.grid(
            which="major",
            axis="x",
            zorder=0,
            color=MOSS,
            linestyle=":",
            dashes=(1, 5),
        )

        if i == 0:
            legend_data = {
                "categories": categories,
                "colors": colors,
                "transpose": transpose,
                "group_variable": group_variable,
                "target_variable": target_variable,
            }

    legend_elements = []
    if legend_data is not None:
        for j, category in enumerate(legend_data["categories"]):
            if not legend_data["transpose"]:
                label = f"{legend_data["target_variable"]} = {category}"
            else:
                label = f"{legend_data["group_variable"]} = {category}"

            legend_elements.append(
                Line2D(
                    [0], [0], color=legend_data["colors"][j],
                    lw=2, label=label
                )
            )

    fig.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(legend_x, legend_y),
        fontsize=10,
    )

    plt.tight_layout()
    plt.show()


def barcharts_by_group(df: pd.DataFrame, target_variables: list) -> None:
    """
    Generates a grid of bar plots, each showing the distribution of a
    target variable.

    Uses predefined color variables in the specified order
    [APPLE, CITRINE, CREAM, NAPLES, MOSS].
    Raises an error if there are more categories than defined colors.

    Args:
            df: pandas DataFrame containing the data.
            target_variables: A list of names of the target variable columns.

    Returns:
            None. Displays the plot or raises an error.
    """

    num_targets = len(target_variables)
    num_cols = 2
    num_rows = (num_targets + 1) // num_cols

    fig = plt.figure(
        figsize=(6 * num_cols, 4 * num_rows),
        dpi=150,
        facecolor=BACKGROUND_COLOR,
    )
    gs = fig.add_gridspec(num_rows, num_cols)
    gs.update(wspace=0.1, hspace=0.1)

    for i, target_variable in enumerate(target_variables):
        row = i // num_cols
        col = i % num_cols
        ax = fig.add_subplot(gs[row, col])

        ax.set_facecolor(BACKGROUND_COLOR)
        ax.tick_params(axis="y", left=False)
        ax.get_yaxis().set_visible(False)
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)

        counts = df[target_variable].value_counts()

        counts = counts.sort_index(ascending=True)

        percentages = (counts / len(df) * 100).round(2)

        colors = [APPLE, CITRINE, CREAM, NAPLES, MOSS, OLIVE, HARVEST, PEAR]

        if len(counts) > len(colors):
            raise ValueError(
                "Not enough colors defined in the palette. "
                "Please add more colors to the `colors` list "
                "to accommodate all categories in your data. "
                f"The variable '{target_variable}' "
                f"has {len(counts)} unique categories, "
                f"but only {len(colors)} colors are defined."
            )

        num_bars = len(counts)
        bar_width = 0.8
        spacing = 0.2 if num_bars > 1 else 0

        total_width = num_bars * bar_width + (num_bars - 1) * spacing
        fig.set_figwidth(max(6 * num_cols, total_width * num_cols))

        positions = [j * (bar_width + spacing) for j in range(num_bars)]

        bars = ax.bar(
            positions,
            counts.values,
            color=colors[:num_bars],
            edgecolor=MOSS,
            linewidth=2,
            width=bar_width,
        )

        for bar, percentage in zip(bars, percentages):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{percentage}%",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(counts.index.astype(str))

        ax.set_xlim(
            -bar_width, (num_bars - 1) * (bar_width + spacing) + bar_width
        )

        ax.set_title(
            f"'{target_variable.replace("_",
			                            " ").title()}' feature Distribution",
            fontsize=14,
            fontweight="bold",
            fontfamily="serif",
        )

    plt.tight_layout()
    plt.show()


def unique_group_counts(
    df: pd.DataFrame, groupby_col: str, count_cols: List[str]
) -> None:
    """Plots unique counts of variables per group,
     cycling through a color palette.

    Args:
            df: The input DataFrame.
            groupby_col: The column to group by.
            count_cols: List of columns to count unique values of.
    """

    unique_values = (
        df.groupby(groupby_col)
        .agg({var: pd.Series.nunique for var in count_cols})
        .reset_index()
    )

    titles = [f"Unique {var}s per {groupby_col}" for var in count_cols]

    num_plots = len(count_cols)
    fig, axes = plt.subplots(
        1, num_plots, figsize=(3 * num_plots, 3), sharey=True
    )

    if num_plots == 1:
        axes = [axes]

    for i, (ax, x, title) in enumerate(zip(axes, count_cols, titles)):
        color_index = i % len(full_pallet)
        sns.countplot(
            data=unique_values,
            x=x,
            ax=ax,
            color=full_pallet[color_index],
            edgecolor=MOSS,
            linewidth=2,
        )

        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_facecolor(BACKGROUND_COLOR)
        ax.grid(False)
        ax.grid(
            which="major",
            axis="y",
            zorder=0,
            color=MOSS,
            linestyle=":",
            dashes=(1, 5),
        )
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()


def binary_heatmap(df: pd.DataFrame, target_col: str, cmap: str = cmap):
    """
    Generates a heatmap of point biserial correlations between a
    target boolean column
    and all other numeric columns in a DataFrame,
    ranked by absolute correlation size.

    Args:
      df: pandas DataFrame containing the data.
      target_col: Name of the boolean target column in the DataFrame.
      cmap: The colormap for the heatmap.

    Returns:
      A matplotlib Axes object containing the heatmap.
    """

    y = df[target_col]
    numeric_df = df.select_dtypes(include=["number"])

    if target_col in numeric_df:
        numeric_df = numeric_df.drop(target_col, axis=1)

    correlations = numeric_df.apply(lambda x: pointbiserialr(x, y)[0])
    correlations = correlations.abs().sort_values(ascending=False)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlations.to_frame(), annot=True, cmap=cmap, vmin=-1, vmax=1
    )
    plt.title(f"Point Biserial Correlation with {target_col} (Ranked)")
    plt.show()


def shapiro_test_by_tag(
    df: pd.DataFrame, tag_col: str, value_col: str, display_output: bool = True
) -> Optional[Dict[str, Tuple[float, float]]]:
    """Shapiro-Wilk tests on value_col grouped by tag_col,
    optionally display."""
    group_true = df.loc[df[tag_col] == True, value_col]
    group_false = df.loc[df[tag_col] == False, value_col]
    stat_true, p_value_true = shapiro(group_true)
    stat_false, p_value_false = shapiro(group_false)

    results = {
        "tag_true": (stat_true, p_value_true),
        "tag_false": (stat_false, p_value_false),
    }

    if display_output:
        display(Markdown("### Shapiro-Wilk Test Results:"))
        for tag, (stat, p_value) in results.items():
            tag_value = "True" if tag == "tag_true" else "False"
            tag_label = f"{tag_col} = {tag_value}"
            display(Markdown(f"**For {tag_label}:**"))
            display(Markdown(f"* Test Statistic: {stat:.3f}"))
            display(Markdown(f"* P-value: {p_value:.3f}"))
        return None
    else:
        return results


def calculate_confidence_interval(data, confidence_level=0.95):
    """Calc. CI for mean of data (Z-interval approx)."""
    mean = data.mean()
    sem = stats.sem(data)
    if pd.isna(sem) or sem == 0:
        return mean, mean, mean
    alpha = 1 - confidence_level
    z_critical = stats.norm.ppf(1 - alpha / 2)
    interval = sem * z_critical
    lower_ci = mean - interval
    upper_ci = mean + interval
    return mean, lower_ci, upper_ci


def non_parametric_test_by_tag(
    df: pd.DataFrame,
    tag_col: str,
    value_col: str,
    confidence_level: float = 0.95,
    display_output: bool = True,
) -> Optional[Dict[str, Dict[str, float]]]:
    """MWU test & CI for value_col grouped by tag_col, optionally display."""
    group_true = df.loc[df[tag_col] == True, value_col]
    group_false = df.loc[df[tag_col] == False, value_col]
    statistic, p_value = stats.mannwhitneyu(group_true, group_false)

    mean_true, lower_ci_true, upper_ci_true = calculate_confidence_interval(
        group_true, confidence_level
    )
    mean_false, lower_ci_false, upper_ci_false = calculate_confidence_interval(
        group_false, confidence_level
    )

    results = {
        "tag_true": {
            "mean": mean_true,
            "lower_ci": lower_ci_true,
            "upper_ci": upper_ci_true,
        },
        "tag_false": {
            "mean": mean_false,
            "lower_ci": lower_ci_false,
            "upper_ci": upper_ci_false,
        },
        "non_parametric_test": {
            "statistic": statistic,
            "p_value": p_value,
        },
    }

    if display_output:
        display(
            Markdown(
                "### Non-parametric Test Results (Mann-Whitney U) "
                "and Confidence Intervals:"
            )
        )
        display(
            Markdown(f"**Confidence Level:** {confidence_level * 100:.0f}%")
        )
        for tag_key, ci_data in results.items():
            if tag_key in ["tag_true", "tag_false"]:
                tag_value = "True" if tag_key == "tag_true" else "False"
                tag_label = f"{tag_col} = {tag_value}"
                display(Markdown(f"**For {tag_label}:**"))
                display(Markdown(f"* Mean Value: {ci_data["mean"]:.2f}"))
                display(
                    Markdown(
                        f"* Confidence Interval: ({ci_data["lower_ci"]:.2f},"
                        f" {ci_data["upper_ci"]:.2f})"
                    )
                )
        np_results = results["non_parametric_test"]
        display(Markdown(f"**Mann-Whitney U Test:**"))
        display(Markdown(f"* p-value = {np_results["p_value"]:.3f}"))
        return None
    else:
        return results


def categorical_feature_tests(
    df: pd.DataFrame,
    cat_col: str,
    target_col: str,
    significance_level: float = 0.05,
    display_output: bool = True,
) -> Optional[Dict[str, float]]:
    """Chi-Squared test & Cramer's V for cat_col vs. binary target_col,
    optionally display."""
    contingency_table = pd.crosstab(df[cat_col], df[target_col])
    chi2_statistic, chi2_p_value, _, _ = stats.chi2_contingency(
        contingency_table
    )

    results = {
        "chi2_statistic": chi2_statistic,
        "chi2_p_value": chi2_p_value,
    }

    if chi2_p_value < significance_level:
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape) - 1
        cramers_v = np.sqrt(chi2_statistic / (n * min_dim))
        results["cramers_v"] = cramers_v

    if display_output:
        display(
            Markdown(
                f"### Categorical Feature Test Results for: `{cat_col}` "
                f"vs `{target_col}`"
            )
        )
        display(Markdown("**Chi-Squared Test:**"))
        display(
            Markdown(
                f"* Chi-Squared Statistic: {results["chi2_statistic"]:.3f}"
            )
        )
        display(Markdown(f"* p-value: {results["chi2_p_value"]:.3f}"))

        if chi2_p_value < significance_level:
            display(
                Markdown(
                    f"**Significant association found "
                    f"(p < {significance_level})**"
                )
            )
            display(
                Markdown(
                    f"**Cramer's V:** {results.get("cramers_v", "N/A"):.3f}"
                )
            )
        else:
            display(
                Markdown(
                    f"**No significant association found "
                    f"(p >= {significance_level})**"
                )
            )

        return None
    else:
        return results


def dynamic_qq(
    data: pd.DataFrame | np.ndarray,
    variable_names: list[str],
    figsize: tuple = None,
    padding: float = 0.05,
) -> None:
    """
    Generate Q-Q plots for multiple variables in a dataset.

    Args:
            data (pd.DataFrame | np.ndarray): Dataset containing the variables
            to plot.
            variable_names (list[str]): List of variable names to plot.
            figsize (tuple, optional): Figure size as (width, height). Default
            is calculated based on plots.
            padding (float, optional): Proportion of padding added to
            axis limits. Default is 0.05.

    Returns:
            None: Displays the Q-Q plots.
    """
    num_plots = len(variable_names)
    num_cols = min(num_plots, 3)
    num_rows = int(np.ceil(num_plots / num_cols))

    if figsize is None:
        figsize = (15, 5 * num_rows)

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=figsize,
        squeeze=False,
        dpi=150,
        facecolor=BACKGROUND_COLOR,
    )
    axes = axes.flatten()

    for i, variable_name in enumerate(variable_names):
        if isinstance(data, pd.DataFrame):
            variable_data = data[variable_name].values
        else:
            variable_data = data[:, i]

        sm.qqplot(variable_data, line="45", fit=True, ax=axes[i])

        axes[i].set_facecolor(BACKGROUND_COLOR)
        axes[i].spines["top"].set_visible(False)
        axes[i].spines["right"].set_visible(False)

        axes[i].get_lines()[0].set_markerfacecolor(CITRINE)
        axes[i].get_lines()[0].set_markeredgecolor(CITRINE)
        axes[i].get_lines()[0].set_alpha(0.7)
        axes[i].get_lines()[0].set_zorder(2)

        for child in axes[i].get_children():
            if (
                isinstance(child, plt.matplotlib.lines.Line2D)
                and child != axes[i].get_lines()[0]
            ):
                child.set_color(MOSS)
                child.set_linestyle("--")
                child.set_alpha(0.8)
                child.set_zorder(1)

        axes[i].set_title(
            f"Q-Q Plot for {variable_name}",
            fontsize=16,
            fontweight="bold",
            fontfamily="serif",
            pad=15,
        )
        axes[i].set_xlabel(
            axes[i].get_xlabel(),
            fontsize=14,
            fontweight="bold",
            fontfamily="serif",
            labelpad=10,
        )
        axes[i].set_ylabel(
            axes[i].get_ylabel(),
            fontsize=14,
            fontweight="bold",
            fontfamily="serif",
            labelpad=10,
        )

        axes[i].tick_params(axis="x", colors="black")
        axes[i].tick_params(axis="y", colors="black")

        axes[i].grid(False)
        axes[i].grid(
            which="major",
            axis="both",
            zorder=0,
            color=MOSS,
            linestyle=":",
            dashes=(1, 5),
        )

        x_min, x_max = axes[i].get_xlim()
        y_min, y_max = axes[i].get_ylim()
        x_range = x_max - x_min
        y_range = y_max - y_min
        axes[i].set_xlim(x_min - x_range * padding, x_max + x_range * padding)
        axes[i].set_ylim(y_min - y_range * padding, y_max + y_range * padding)

    for i in range(num_plots, len(axes)):
        axes[i].axis("off")

        sns.despine()

    fig.tight_layout()
    plt.show()


def dynamic_confusion_matrix(
    models: List, X: pd.DataFrame, y: pd.Series, display_labels: List[str]
):
    """
    Plots multiple confusion matrices for a given list of models.

    Args:
      models: A list of trained machine learning models.
      X: The feature matrix as a Pandas DataFrame.
      y: The target variable array as a Pandas Series.
      display_labels: A list of string labels for the classes in
                                      the confusion matrix.
    """
    num_models = len(models)
    fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 5))

    if num_models == 1:
        axes = [axes]

    for i, model in enumerate(models):
        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=display_labels
        )
        disp.plot(ax=axes[i], cmap=cmap)
        axes[i].set_title(
            model.__class__.__name__, fontsize=14, fontweight="bold"
        )
        for text in disp.text_.ravel():
            text.set_fontsize(18)
            text.set_fontweight("bold")

    plt.tight_layout()
    plt.show()


def optimize_and_evaluate_classifiers(
    classifiers: Dict[str, ClassifierMixin],
    grid: Dict[str, Dict[str, Any]],
    X: pd.DataFrame,
    y: pd.Series,
    n_iter_random_search: int = 10,
    cv_folds: int = 5,
    scoring_metric: str = "accuracy",
    random_state: int = 42,
) -> Tuple[
    pd.DataFrame, Dict[str, Dict[str, Any]], Dict[str, ClassifierMixin]
]:
    """
    Perform random search hyperparameter optimization for multiple classifiers.

    Args:
            classifiers (Dict): Classifier names and objects.
            grid (Dict): Hyperparameter grids for each classifier.
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target variable.
            n_iter_random_search (int): Number of random search iterations.
            Default is 10.
            cv_folds (int): Number of cross-validation folds. Default is 10.
            scoring_metric (str): Scoring metric for evaluation.
            Default is "accuracy".
            random_state (int): Random state for reproducibility. Default is 42.

    Returns:
            Tuple:
                    - Validation scores DataFrame
                     (classifier names,validation accuracy, and timing).
                    - Best hyperparameters for each classifier.
                    - Best estimators after fitting the best parameters.
    """
    validation_scores = pd.DataFrame(
        columns=[
            "Classifier",
            "Average Validation Accuracy (%)",
            "Total Time (s)",
        ]
    )
    clf_best_params: Dict[str, Dict[str, Any]] = {}
    best_estimators: Dict[str, ClassifierMixin] = {}

    cv = StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=random_state
    )

    print(
        f"{"Model":<15} {"Average Validation Accuracy (%)":>35} "
        f"{"Total Time (s)":>15}  {"Best Params"}"
    )
    print("=" * 120)

    main_pbar = tqdm(
        total=len(classifiers), desc="Optimizing Classifiers", position=0
    )

    for idx, (name, classifier) in enumerate(classifiers.items()):
        start_time = time.time()

        param_grid = grid[name]
        param_sampler = ParameterSampler(
            param_grid, n_iter=n_iter_random_search, random_state=random_state
        )
        best_accuracy = -np.inf
        current_best_params = None

        with tqdm(
            total=n_iter_random_search,
            desc=f"{name}",
            position=idx + 1,
            leave=False,
        ) as inner_pbar:
            for params in param_sampler:
                classifier.set_params(**params)
                scores = cross_val_score(
                    classifier,
                    X,
                    y,
                    cv=cv,
                    scoring=scoring_metric,
                    n_jobs=-1,
                    error_score="raise",
                )
                avg_accuracy = np.mean(scores)

                if avg_accuracy > best_accuracy:
                    best_accuracy = avg_accuracy
                    current_best_params = params

                inner_pbar.update(1)

        total_time = time.time() - start_time

        clf_best_params[name] = current_best_params
        best_estimators[name] = classifier.set_params(**current_best_params)

        new_row = pd.DataFrame(
            {
                "Classifier": [name],
                "Average Validation Accuracy (%)": [
                    np.round(100 * best_accuracy, 2)
                ],
                "Total Time (s)": [np.round(total_time, 2)],
            }
        )
        validation_scores = pd.concat(
            [validation_scores, new_row], ignore_index=True
        )

        print(
            f"{name:<15} {np.round(100 * best_accuracy, 2):>35.2f}%"
            f"{np.round(total_time, 2):>15.2f}s"
            f"{clf_best_params[name]}"
        )

        main_pbar.update(1)

    main_pbar.close()

    return validation_scores, clf_best_params, best_estimators


def plot_model_evaluation(
    y_true: np.ndarray,
    y_proba: np.ndarray = None,
    classifier=None,
    X: np.ndarray = None,
    figsize: tuple = (9, 4),
    padding: float = 0.05,
) -> None:
    """
    Plot ROC and Precision-Recall curves for model evaluation.

    Args:
            y_true (np.ndarray): True binary labels.
            y_proba (np.ndarray, optional): Probability estimates of
            the positive class.
            If None, computed using the classifier.
            classifier (object, optional): Fitted classifier with
            `predict_proba` method.
            Required if `y_proba` is None.
            X (np.ndarray, optional): Feature data for predictions.
            Required if `y_proba` is None.
            figsize (tuple, optional): Figure size as (width, height).
            Default is (10, 4).
            padding (float, optional): Proportion of padding added to axis
            limits.
            Default is 0.05.

    Returns:
            None: Displays the plots.
    """

    if y_proba is None:
        if classifier is None or X is None:
            raise ValueError(
                "If y_proba is None, both classifier and X must be provided"
            )
        y_proba = classifier.predict_proba(X)[:, 1]

    fig, ax = plt.subplots(
        1, 2, figsize=figsize, dpi=150, facecolor=BACKGROUND_COLOR
    )

    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    ax[0].plot(
        fpr,
        tpr,
        color=ml_colors[0],
        alpha=0.7,
        linewidth=2,
        zorder=2,
        label=f"AUC: {roc_auc:.2f}",
    )
    ax[0].plot([0, 1], [0, 1], color=MOSS, linestyle="--", alpha=0.8, zorder=1)

    ax[0].set_facecolor(BACKGROUND_COLOR)
    ax[0].spines["top"].set_visible(False)
    ax[0].spines["right"].set_visible(False)

    x_range_roc = fpr.max() - fpr.min()
    y_range_roc = tpr.max() - tpr.min()
    ax[0].set_xlim(
        fpr.min() - x_range_roc * padding, fpr.max() + x_range_roc * padding
    )
    ax[0].set_ylim(
        tpr.min() - y_range_roc * padding, tpr.max() + y_range_roc * padding
    )

    ax[0].grid(
        which="major",
        axis="both",
        zorder=0,
        color=MOSS,
        linestyle=":",
        dashes=(1, 5),
    )

    ax[0].set_xlabel(
        "False Positive Rate",
        fontsize=14,
        fontweight="bold",
        fontfamily="serif",
        labelpad=10,
    )
    ax[0].set_ylabel(
        "True Positive Rate",
        fontsize=14,
        fontweight="bold",
        fontfamily="serif",
        labelpad=10,
    )
    ax[0].set_title(
        "ROC Curve", fontsize=16, fontweight="bold", fontfamily="serif", pad=15
    )

    ax[0].tick_params(axis="x", colors="black")
    ax[0].tick_params(axis="y", colors="black")

    legend = ax[0].legend(loc="lower right", facecolor=BACKGROUND_COLOR)
    legend.get_frame().set_edgecolor(MOSS)

    prec, recall, _ = precision_recall_curve(y_true, y_proba)

    ax[1].plot(
        recall, prec, color=ml_colors[0], alpha=0.7, linewidth=2, zorder=2
    )

    ax[1].set_facecolor(BACKGROUND_COLOR)
    ax[1].spines["top"].set_visible(False)
    ax[1].spines["right"].set_visible(False)

    x_range_pr = recall.max() - recall.min()
    y_range_pr = prec.max() - prec.min()
    ax[1].set_xlim(
        recall.min() - x_range_pr * padding,
        recall.max() + x_range_pr * padding,
    )
    ax[1].set_ylim(
        prec.min() - y_range_pr * padding, prec.max() + y_range_pr * padding
    )

    ax[1].grid(
        which="major",
        axis="both",
        zorder=0,
        color=MOSS,
        linestyle=":",
        dashes=(1, 5),
    )

    ax[1].set_xlabel(
        "Recall",
        fontsize=14,
        fontweight="bold",
        fontfamily="serif",
        labelpad=10,
    )
    ax[1].set_ylabel(
        "Precision",
        fontsize=14,
        fontweight="bold",
        fontfamily="serif",
        labelpad=10,
    )
    ax[1].set_title(
        "Precision-Recall Curve",
        fontsize=16,
        fontweight="bold",
        fontfamily="serif",
        pad=15,
    )

    ax[1].tick_params(axis="x", colors="black")
    ax[1].tick_params(axis="y", colors="black")

    sns.despine()

    fig.tight_layout()
    plt.show()


def shap_summary(
    model: BaseEstimator,
    X: np.ndarray,
    feature_names: List[str],
    model_name: Optional[str] = "Model",
    figsize: tuple = (9, 4),
    padding: float = 0.05,
) -> None:
    """
    Generate a SHAP summary plot with consistent evaluation style.

    Args:
            model (BaseEstimator): Trained model compatible with SHAP
            X (np.ndarray): Feature matrix for SHAP computation
            feature_names (List[str]): Feature names for labeling
            model_name (str, optional): Model name for title. Default "Model"
            figsize (tuple): Figure dimensions. Default (9, 4)
            padding (float): Axis padding proportion. Default 0.05

    Returns:
            None: Displays styled SHAP summary plot
    """
    fig = plt.figure(figsize=figsize, facecolor=BACKGROUND_COLOR, dpi=150)
    ax = fig.gca()

    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap.summary_plot(
        shap_values, X, feature_names=feature_names, show=False, cmap=cmap
    )

    ax.set_facecolor(BACKGROUND_COLOR)
    ax.set_xlabel(
        "Feature Impact",
        fontsize=14,
        fontweight="bold",
        fontfamily="serif",
        labelpad=10,
    )
    ax.set_title(
        f"SHAP Summary - {model_name}",
        fontsize=16,
        fontweight="bold",
        fontfamily="serif",
        pad=15,
    )

    x_min, x_max = ax.get_xlim()
    x_range = x_max - x_min
    ax.set_xlim(x_min - x_range * padding, x_max + x_range * padding)

    ax.tick_params(axis="both", which="major", labelsize=12, colors="black")
    for spine in ax.spines.values():
        spine.set_color(MOSS)
        spine.set_linewidth(0.8)

    ax.grid(
        True,
        which="major",
        axis="both",
        zorder=0,
        color=MOSS,
        linestyle=":",
        dashes=(1, 5),
        alpha=0.7,
    )

    cbar = fig.axes[-1]
    cbar.set_ylabel(
        "Feature Value",
        fontsize=12,
        fontfamily="serif",
        fontweight="bold",
        color=MOSS,
        labelpad=10,
    )
    cbar.tick_params(labelsize=10, colors=MOSS)
    cbar.spines[:].set_color(MOSS)
    cbar.set_facecolor(BACKGROUND_COLOR)

    plt.tight_layout()
    plt.show()


def feature_importances(
    model: BaseEstimator,
    feature_names: List[str],
    top_n: int = 10,
    figsize: Tuple[int, int] = (9, 4),
    model_name: str = "Model",
    padding: float = 0.05,
) -> None:
    """Plot styled top/bottom feature importances for a model.

    Args:
            model: Fitted classifier with feature_importances_.
            feature_names: Feature names.
            top_n: Number of top/bottom features to show.
            figsize: Plot size (width, height) in inches.
            model_name: Model name for plot title.
            padding: Proportional padding for axis limits.
    Returns:
            None: Displays feature importance plots.
    """
    importances = model.feature_importances_

    importances_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values(by="Importance", ascending=False)

    top_features = importances_df.head(top_n)
    bottom_features = importances_df.tail(top_n).iloc[::-1]

    fig, axes = plt.subplots(
        1, 2, figsize=figsize, facecolor=BACKGROUND_COLOR, dpi=100
    )

    sns.barplot(
        x="Importance",
        y="Feature",
        data=top_features,
        palette=full_pallet,
        ax=axes[0],
    )
    axes[0].set_title(
        f"{model_name} - Top {top_n} Features",
        fontsize=14,
        fontweight="bold",
        fontfamily="serif",
    )
    axes[0].set_xlabel(
        "Importance", fontsize=12, fontweight="bold", fontfamily="serif"
    )
    axes[0].set_ylabel("", fontsize=12, fontweight="bold", fontfamily="serif")
    axes[0].set_facecolor(BACKGROUND_COLOR)

    x_min_top, x_max_top = axes[0].get_xlim()
    x_range_top = x_max_top - x_min_top
    axes[0].set_xlim(
        x_min_top - x_range_top * padding, x_max_top + x_range_top * padding
    )

    axes[0].tick_params(axis="x", colors="black")
    axes[0].tick_params(axis="y", colors="black")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].grid(axis="x", color=MOSS, linestyle="--")

    sns.barplot(
        x="Importance",
        y="Feature",
        data=bottom_features,
        palette=full_pallet[::-1],
        ax=axes[1],
    )
    axes[1].set_title(
        f"{model_name} - Bottom {top_n} Features",
        fontsize=14,
        fontweight="bold",
        fontfamily="serif",
    )
    axes[1].set_xlabel(
        "Importance", fontsize=12, fontweight="bold", fontfamily="serif"
    )
    axes[1].set_ylabel("", fontsize=12, fontweight="bold", fontfamily="serif")
    axes[1].set_facecolor(BACKGROUND_COLOR)

    x_min_bottom, x_max_bottom = axes[1].get_xlim()
    x_range_bottom = x_max_bottom - x_min_bottom
    axes[1].set_xlim(
        x_min_bottom - x_range_bottom * padding,
        x_max_bottom + x_range_bottom * padding,
    )

    axes[1].tick_params(axis="x", colors="black")
    axes[1].tick_params(axis="y", colors="black")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].grid(axis="x", color=MOSS, linestyle="--")

    fig.tight_layout()
    plt.show()
