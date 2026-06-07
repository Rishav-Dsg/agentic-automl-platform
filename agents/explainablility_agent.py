import shap
import numpy as np
from typing import cast

class ExplainabilityAgent:

    def __init__(self, model, X):
        self.model = model
        self.X = X

    def explain(self):
        if hasattr(self.model, "feature_importances_"):
            importance = (
                self.model.feature_importances_
            )

            feature_scores = {
                col: float(score)
                for col, score in zip(
                    self.X.columns,
                    importance
                )
            }

            sorted_features = sorted(
                feature_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )

            return {
                "feature_importance":
                    feature_scores,

                "top_features":
                    sorted_features[:10]
            }

    # fallback to SHAP
        explainer = shap.Explainer(self.model,self.X)

        shap_values = explainer(self.X,check_additivity=False)

        if isinstance(shap_values, list):
            values = shap_values[0].values
        else:
            values = shap_values.values
        
        values = cast(np.ndarray, values)
        if values.ndim == 3:
            values = np.abs(values).mean(axis=2)
        importance = np.abs(values).mean(axis=0)

        feature_scores = {
            col: float(score)
            for col, score in zip(self.X.columns,importance)
        }

        sorted_features = sorted(
            feature_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return {
            "feature_importance":
                feature_scores,
            "top_features":
                sorted_features[:10]
        }