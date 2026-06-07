class EvaluationAgent:
    def __init__(self, results):
        self.results = results

    def select_best_model(self):
        best_model = max(
            self.results,
            key=self.results.get
        )

        best_score = self.results[best_model]
        return best_model, best_score

    def generate_reasoning(
        self,
        best_model,
        profile,
        task_type
    ):
        reasoning = []
        reasoning.append(
            f"Selected {best_model}"
        )
        reasoning.append(
            f"Task type detected: {task_type}"
        )
        reasoning.append(
            f"Dataset size: "
            f"{profile['dataset_size']}"
        )
        if (
            len(profile["numerical_columns"])
            >
            len(profile["categorical_columns"])
        ):
            reasoning.append(
                "Dataset is primarily numerical"
            )

        if "random_forest" in best_model:

            reasoning.append(
                "Random Forest performs well "
                "on nonlinear tabular datasets"
            )

        elif "xgboost" in best_model:

            reasoning.append(
                "XGBoost handles complex "
                "feature interactions effectively"
            )

        elif "logistic" in best_model:

            reasoning.append(
                "Logistic Regression provides "
                "strong linear baseline performance"
            )

        return reasoning