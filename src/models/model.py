import numpy as np
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder


def as_category(df):
    """LightGBM only handles 'category' columns natively as categorical."""
    return df.astype("category")


def build_model_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    n_trees: int = 1500,
    learning_rate: float = 0.03,
    num_leaves: int = 31,
) -> Pipeline:
    """
    Building the pipeline model
    """

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
            ("as_category", FunctionTransformer(as_category)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("Preprocessing numerical", numeric_transformer, numeric_features),
            ("Preprocessing categorical", categorical_transformer, categorical_features),
        ]
    ).set_output(transform="pandas")

    pipe_lgbm = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "regressor",
                LGBMRegressor(
                    n_estimators=n_trees,
                    learning_rate=learning_rate,
                    num_leaves=num_leaves,
                    min_child_samples=10,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=0,
                    verbose=-1,
                ),
            ),
        ]
    )

    return TransformedTargetRegressor(
        regressor=pipe_lgbm, func=np.log1p, inverse_func=np.expm1
    )

