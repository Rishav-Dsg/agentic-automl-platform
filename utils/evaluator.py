from sklearn.model_selection import cross_val_score

class Evaluator:

    def __init__(self, cv=5, scoring="accuracy"):
        self.cv = cv
        self.scoring = scoring   # FIX: caller now sets scoring ("accuracy" or "r2")

    def evaluate(self, model, X, y):
        scores = cross_val_score(
            model, X, y,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=-1
        )
        return {
            "mean_score": scores.mean(),
            "std_score":  scores.std(),
            "all_scores": scores
        }
    