class RoutingAgent:

    def __init__(self, profile, task_type):

        self.profile = profile
        self.task_type = task_type

    def select_models(self):

        selected_models = []

        numerical_count = len(
            self.profile["numerical_columns"]
        )

        categorical_count = len(
            self.profile["categorical_columns"]
        )

        dataset_size = self.profile["dataset_size"]

        if self.task_type == "classification":

            selected_models.append(
                "logistic_regression"
            )

            if numerical_count > 0:

                selected_models.append(
                    "random_forest_classifier"
                )

                selected_models.append(
                    "xgboost_classifier"
                )

            if dataset_size == "small":

                selected_models.append(
                    "svm"
                )

        elif self.task_type == "regression":

            selected_models.append(
                "linear_regression"
            )

            selected_models.append(
                "random_forest_regressor"
            )

            selected_models.append(
                "xgboost_regressor"
            )

        return selected_models