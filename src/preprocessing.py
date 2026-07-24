from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def create_preprocessor(X_train):
    """
    Create a preprocessing pipeline for numerical and categorical features.
    """

    categorical_features = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_features = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    numeric_transformer = Pipeline([
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    return preprocessor

# from src.preprocessing import create_preprocessor

# preprocessor = create_preprocessor(X_train)