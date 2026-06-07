from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import (
accuracy_score,
precision_score,
recall_score,
f1_score,
confusion_matrix,
mean_squared_error,
mean_absolute_error,
r2_score
)
from sklearn.impute import SimpleImputer

import numpy as np
import time

from utils.model_registry import MODEL_REGISTRY
from utils.evaluator import Evaluator

class TrainingAgent:

    def __init__(
        self,
        df,
        target_column,
        models,
        task_type="classification"
    ):
        self.df = df
        self.target_column = target_column
        self.models = models
        self.task_type = task_type
        self.feature_names = []

    def prepare_data(self):

        X = self.df.drop(
            columns=[self.target_column]
        )

        X = X.replace(["?", "", "NA", "N/A", "null", "None"], np.nan)

        num_cols = X.select_dtypes(include=["number"]).columns

        if len(num_cols):
            imputer = SimpleImputer(
                strategy="median"
            )
            X[num_cols] = imputer.fit_transform(X[num_cols])

        cat_cols = X.select_dtypes(include=["object", "category"]).columns

        if len(cat_cols):
            imputer = SimpleImputer(
                strategy="most_frequent"
            )
            X[cat_cols] = imputer.fit_transform(X[cat_cols])

        self.feature_names = X.columns.tolist()

        id_cols = [
            col
            for col in X.columns
            if col.lower() in (
                "id",
                "index"
            )
        ]

        if id_cols:
            X = X.drop(
                columns=id_cols
            )

        y = self.df[
            self.target_column
        ]

        cat_cols = X.select_dtypes(
            include=[
                "object",
                "category"
            ]
        ).columns.tolist()

        if cat_cols:

            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )

            X = X.copy()

            X[cat_cols] = encoder.fit_transform(
                X[cat_cols]
            )

        if self.task_type == "classification":
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)

        else:
            y = y.values.astype(float)

        X = X.fillna(0)
        print("\n===== DATA CHECK =====")
        print("Remaining NaNs:", X.isna().sum().sum())
        if X.isna().sum().sum():
            print(
                X.isna().sum()[
                    X.isna().sum() > 0
                ]
            )
        print("======================\n")
        print(X.dtypes)
        print(X.head())

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=(
                y
                if self.task_type == "classification"
                else None
            )
        )


        return (
            X_train,
            X_test,
            y_train,
            y_test
        )

    def get_model_instance(
        self,
        model_name
    ):

        if model_name not in MODEL_REGISTRY:

            raise ValueError(
                f"Unknown model: {model_name}"
            )

        return MODEL_REGISTRY[
            model_name
        ]["model"]()

    def train_models(self):

        (
            X_train,
            X_test,
            y_train,
            y_test
        ) = self.prepare_data()

        scoring = (
            "accuracy"
            if self.task_type == "classification"
            else "r2"
        )

        evaluator = Evaluator(
            scoring=scoring
        )

        results = {}
        trained_models = {}
        detailed_metrics = {}

        for model_name in self.models:

            print(
                f"\nTraining {model_name}..."
            )

            model = self.get_model_instance(
                model_name
            )

            evaluation = evaluator.evaluate(
                model,
                X_train.fillna(0),
                y_train
            )

            mean_score = evaluation["mean_score"]

            start = time.time()

            model.fit(X_train,y_train)

            training_time = (
                time.time() - start
            )

            predictions = model.predict(
                X_test
            )

            results[
                model_name
            ] = mean_score

            trained_models[
                model_name
            ] = model

            if self.task_type == "classification":

                detailed_metrics[
                    model_name
                ] = {

                    "accuracy":
                        accuracy_score(
                            y_test,
                            predictions
                        ),

                    "precision":
                        precision_score(
                            y_test,
                            predictions,
                            average="weighted",
                            zero_division=0
                        ),

                    "recall":
                        recall_score(
                            y_test,
                            predictions,
                            average="weighted",
                            zero_division=0
                        ),

                    "f1_score":
                        f1_score(
                            y_test,
                            predictions,
                            average="weighted",
                            zero_division=0
                        ),

                    "confusion_matrix":
                        confusion_matrix(
                            y_test,
                            predictions
                        ),

                    "training_time":
                        training_time,

                    "supports_probability":
                        hasattr(
                            model,
                            "predict_proba"
                        )
                }

            else:

                mse = mean_squared_error(
                    y_test,
                    predictions
                )

                detailed_metrics[
                    model_name
                ] = {

                    "r2":
                        r2_score(
                            y_test,
                            predictions
                        ),

                    "mae":
                        mean_absolute_error(
                            y_test,
                            predictions
                        ),

                    "mse":
                        mse,

                    "rmse":
                        np.sqrt(mse),

                    "training_time":
                        training_time,

                    "supports_probability":
                        False
                }

            print(
                f"{model_name} CV {scoring}: "
                f"{mean_score:.4f}"
            )
 
        return (
            results,
            trained_models,
            detailed_metrics,
            self.feature_names
        )
