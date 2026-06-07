import optuna
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier, XGBRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)  # reduce noise


class TuningAgent:
    def __init__(self, X, y, best_model_name="random_forest_classifier", task_type="classification"):
        self.X = X
        self.y = y
        self.best_model_name = best_model_name
        self.task_type = task_type
        self.scoring = "accuracy" if task_type == "classification" else "r2"

    # ------------------------------------------------------------------
    # Public entry point  –  dispatches to the right tuner
    # ------------------------------------------------------------------

    def tune_best_model(self):
        """FIX: tune whichever model actually won, not always RandomForest."""
        name = self.best_model_name.lower()

        if "random_forest" in name:
            return self._tune_random_forest()
        elif "xgboost" in name:
            return self._tune_xgboost()
        elif "logistic" in name:
            return self._tune_logistic_regression()
        elif "linear_regression" in name:
            # LinearRegression has no meaningful hyperparameters to tune
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            score = cross_val_score(model, self.X, self.y, cv=5, scoring=self.scoring).mean()
            return {}, score
        elif "svm" in name:
            return self._tune_svm()
        else:
            # Fallback: tune random forest as a strong baseline
            return self._tune_random_forest()

    # ------------------------------------------------------------------
    # Individual tuners
    # ------------------------------------------------------------------

    def _tune_random_forest(self):
        is_clf = self.task_type == "classification"

        def objective(trial):
            params = {
                "n_estimators":     trial.suggest_int("n_estimators", 50, 300),
                "max_depth":        trial.suggest_int("max_depth", 2, 20),
                "min_samples_split":trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features":     trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                "random_state": 42
            }
            model = (RandomForestClassifier(**params) if is_clf
                     else RandomForestRegressor(**params))
            return cross_val_score(model, self.X, self.y, cv=5, scoring=self.scoring).mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20)
        return study.best_params, study.best_value

    def _tune_xgboost(self):
        is_clf = self.task_type == "classification"

        def objective(trial):
            params = {
                "n_estimators":  trial.suggest_int("n_estimators", 50, 300),
                "max_depth":     trial.suggest_int("max_depth", 2, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample":     trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "random_state": 42
            }
            if is_clf:
                params["eval_metric"] = "mlogloss"
                model = XGBClassifier(**params)
            else:
                model = XGBRegressor(**params)
            return cross_val_score(model, self.X, self.y, cv=5, scoring=self.scoring).mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20)
        return study.best_params, study.best_value

    def _tune_logistic_regression(self):
        def objective(trial):
            C       = trial.suggest_float("C", 1e-3, 100.0, log=True)
            solver  = trial.suggest_categorical("solver", ["lbfgs", "saga"])
            penalty = "l2"
            model = LogisticRegression(
                C=C, solver=solver, penalty=penalty,
                max_iter=1000, random_state=42
            )
            return cross_val_score(model, self.X, self.y, cv=5, scoring=self.scoring).mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20)
        return study.best_params, study.best_value

    def _tune_svm(self):
        def objective(trial):
            C      = trial.suggest_float("C", 1e-2, 100.0, log=True)
            kernel = trial.suggest_categorical("kernel", ["rbf", "linear", "poly"])
            gamma  = trial.suggest_categorical("gamma", ["scale", "auto"])
            model  = SVC(C=C, kernel=kernel, gamma=gamma, random_state=42)
            return cross_val_score(model, self.X, self.y, cv=5, scoring=self.scoring).mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20)
        return study.best_params, study.best_value

    # ------------------------------------------------------------------
    # Legacy shim kept for backward compatibility
    # ------------------------------------------------------------------

    def tune_random_forest(self):
        return self._tune_random_forest()