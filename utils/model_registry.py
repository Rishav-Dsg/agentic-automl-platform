from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import VotingClassifier

MODEL_REGISTRY = {

    # ------------------------------------------------------------------ #
    #  Classification                                                      #
    # ------------------------------------------------------------------ #

    "logistic_regression": {
        "task": "classification",
        "family": "linear",
        "model": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42))
        ])
    },

    "random_forest_classifier": {
        "task": "classification",
        "family": "tree_ensemble",
        "model": lambda: RandomForestClassifier(random_state=42)
    },

    "xgboost_classifier": {
        "task": "classification",
        "family": "boosting",
        "model": lambda: XGBClassifier(
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0
        )
    },

    "svm": {
        "task": "classification",
        "family": "kernel",
        # FIX: wrap SVM in a scaler pipeline — SVM is scale-sensitive
        "model": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(probability=True, random_state=42))
        ])
    },

    # ------------------------------------------------------------------ #
    #  Regression                                                          #
    # ------------------------------------------------------------------ #

    "linear_regression": {
        "task": "regression",
        "family": "linear",
        # FIX: use Ridge instead of plain LinearRegression for robustness
        "model": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ])
    },

    "random_forest_regressor": {
        "task": "regression",
        "family": "tree_ensemble",
        "model": lambda: RandomForestRegressor(random_state=42)
    },

    "xgboost_regressor": {
        "task": "regression",
        "family": "boosting",
        "model": lambda: XGBRegressor(
            random_state=42,
            verbosity=0
        )
    },
}