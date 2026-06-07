from agents.profiler_agent import ProfilerAgent
from agents.task_agent import TaskAgent
from agents.routing_agent import RoutingAgent
from agents.training_agent import TrainingAgent
from agents.evaluation_agent import EvaluationAgent
from agents.tuning_agent import TuningAgent
from agents.data_quality_agent import DataQualityAgent
from agents.llm_critic_agent import LLMCriticAgent
from agents.explainablility_agent import ExplainabilityAgent

from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

import numpy as np


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------

def profiling_node(state):
    profiler = ProfilerAgent(state["dataset_path"])
    profile  = profiler.profile_dataset()
    return {"profile": profile, "processed_df": profiler.df}


# ---------------------------------------------------------------------------
# Task detection
# ---------------------------------------------------------------------------

def task_node(state):
    df         = state["processed_df"]
    task_agent = TaskAgent(df)
    targets    = task_agent.detect_target_column()
    target     = targets[0]
    task_type  = task_agent.detect_task_type(target)
    return {"target_column": target, "task_type": task_type}


# ---------------------------------------------------------------------------
# Model selection  (replaces routing_node)
# ---------------------------------------------------------------------------

def model_selection_node(state):
    profile   = state["profile"]
    rows      = profile["rows"]
    numerical = len(profile["numerical_columns"])
    categorical = len(profile["categorical_columns"])
    models    = []
    reasoning = []

    if state["task_type"] == "classification":
        models.append("logistic_regression")
        models.append("random_forest_classifier")
        reasoning.append("Selected Random Forest for nonlinear patterns.")

        if rows > 1000:
            models.append("xgboost_classifier")
            reasoning.append("Selected XGBoost for medium dataset.")

        if categorical > numerical:
            # CatBoost not in registry; fall back to xgboost if not already added
            if "xgboost_classifier" not in models:
                models.append("xgboost_classifier")
            reasoning.append("High categorical ratio: using XGBoost.")
    else:
        models.extend(["linear_regression", "random_forest_regressor", "xgboost_regressor"])
        reasoning.append("Selected regression models.")

    return {"selected_models": models, "model_selection_reasoning": reasoning}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def training_node(state):
    df             = state["processed_df"]
    training_agent = TrainingAgent(
        df, state["target_column"], state["selected_models"], state["task_type"]
    )
    results, trained_models, detailed_metrics, feature_names = training_agent.train_models()
    return {
        "training_results": results,
        "trained_models":   trained_models,
        "detailed_metrics": detailed_metrics,
        "feature_names":    feature_names,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluation_node(state):
    evaluation_agent       = EvaluationAgent(state["training_results"])
    best_model, best_score = evaluation_agent.select_best_model()
    reasoning              = evaluation_agent.generate_reasoning(
        best_model, state["profile"], state["task_type"]
    )
    return {"best_model": best_model, "best_score": best_score, "reasoning": reasoning}


# ---------------------------------------------------------------------------
# Experiment tracker  – only tracks individual model results
# ---------------------------------------------------------------------------

def experiment_tracker_node(state):
    history = state.get("experiment_history", [])
    history = list(history)   # copy so we don't mutate shared state

    history.append({
        "iteration": len(history) + 1,
        "model":     state["best_model"],
        "score":     state["best_score"],
    })

    # Track global best across iterations (individual models only)
    best_entry = max(history, key=lambda x: x["score"])
    return {
        "experiment_history":  history,
        "global_best_score":   best_entry["score"],
        "global_best_model":   best_entry["model"],
    }


# ---------------------------------------------------------------------------
# Ensemble  – FIX: fit a fresh VotingClassifier instead of reusing stale ones
# ---------------------------------------------------------------------------

def ensemble_node(state):
    """
    Build and fit a soft-voting ensemble from the trained individual models.
    Only runs for classification and only when both RF and XGB are present.
    """
    if state["task_type"] != "classification":
        return {}

    trained_models = state["trained_models"]
    required = {"random_forest_classifier", "xgboost_classifier"}
    if not required.issubset(trained_models.keys()):
        return {}

    # FIX: build a fresh VotingClassifier with the already-fitted estimators.
    # VotingClassifier's predict/predict_proba works fine with pre-fitted
    # estimators when we set them directly (sklearn >= 1.0 supports this via
    # the estimators_ attribute after a dummy fit, but the cleanest approach
    # is to just fit it on the training data again using prepare_data).
    df     = state["processed_df"]
    target = state["target_column"]
    training_agent = TrainingAgent(
        df, target, list(trained_models.keys()), state["task_type"]
    )
    X_train, _, y_train, _ = training_agent.prepare_data()

    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    # Fresh (unfitted) instances so VotingClassifier can fit them properly
    ensemble = VotingClassifier(
        estimators=[
            ("rf",  RandomForestClassifier(random_state=42)),
            ("xgb", XGBClassifier(eval_metric="mlogloss", random_state=42, verbosity=0)),
        ],
        voting="soft",
    )
    ensemble.fit(X_train, y_train)
    return {"ensemble_model": ensemble}


# ---------------------------------------------------------------------------
# Ensemble evaluation  – FIX: encode y, keep separate from main history
# ---------------------------------------------------------------------------

def ensemble_evaluation_node(state):
    if "ensemble_model" not in state or state.get("ensemble_model") is None:
        return {"ensemble_score": 0.0}

    df     = state["processed_df"]
    target = state["target_column"]

    # Use the same prepare_data() pipeline so label encoding is consistent
    training_agent = TrainingAgent(
        df, target, state["selected_models"], state["task_type"]
    )
    X_train, X_test, y_train, y_test = training_agent.prepare_data()

    # Combine back for full-dataset CV
    import numpy as np
    import pandas as pd
    X_full = pd.concat([X_train, X_test])
    y_full = np.concatenate([y_train, y_test])

    model  = state["ensemble_model"]
    scores = cross_val_score(model, X_full, y_full, cv=5, scoring="accuracy")
    ensemble_score = scores.mean()

    print(f"\nEnsemble CV accuracy: {ensemble_score:.4f}")
    return {"ensemble_score": ensemble_score}

# ---------------------------------------------------------------------------
# Tuning  – tunes the actual best model
# ---------------------------------------------------------------------------

def tuning_node(state):
    df             = state["processed_df"]
    training_agent = TrainingAgent(
        df, state["target_column"], state["selected_models"], state["task_type"]
    )
    X_train, _, y_train, _ = training_agent.prepare_data()

    tuning_agent = TuningAgent(X_train, y_train, state["best_model"], state["task_type"])
    best_params, tuned_score = tuning_agent.tune_best_model()

    return {"tuning_results": {"best_params": best_params, "tuned_score": tuned_score}}


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def feature_engineering_node(state):

    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import PolynomialFeatures

    df = state["processed_df"].copy()

    target = state["target_column"]

    if state["retry_count"] >= 2:

        return {
            "processed_df": df,
            "retry_count": state["retry_count"]
        }

    cols_to_drop = []

    for col in df.columns:

        if col == target:
            continue

        if df[col].nunique() <= 1:
            cols_to_drop.append(col)

        elif (
            df[col].dtype == object
            and
            df[col].nunique() / len(df) > 0.95
        ):
            cols_to_drop.append(col)

        elif "id" in col.lower():
            cols_to_drop.append(col)

    if cols_to_drop:

        df = df.drop(
            columns=cols_to_drop
        )

    numeric_cols = [

        c

        for c in df.select_dtypes(
            include=np.number
        ).columns

        if c != target
    ]

    feature_scores = {}

    if target in df.columns:

        for col in numeric_cols:

            try:

                corr = abs(
                    df[col].corr(
                        df[target]
                    )
                )

                feature_scores[col] = float(corr)

            except Exception:
                pass

    for col in numeric_cols:

        try:

            if (
                df[col].skew() > 1
                and
                df[col].min() >= 0
            ):

                df[col] = np.log1p(
                    df[col]
                )

        except Exception:
            pass

    if len(feature_scores) >= 2:

        top_features = sorted(
            feature_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        top_cols = [
            x[0]
            for x in top_features
        ]

        poly = PolynomialFeatures(
            degree=2,
            interaction_only=True,
            include_bias=False
        )

        poly_data = poly.fit_transform(
            df[top_cols]
        )

        poly_names = poly.get_feature_names_out(
            top_cols
        )

        interaction_cols = [
            i
            for i, name in enumerate(poly_names)
            if " " in name
        ]

        poly_df = pd.DataFrame(
            poly_data[:, interaction_cols],
            columns=[
                poly_names[i].replace(
                    " ",
                    "_x_"
                )
                for i in interaction_cols
            ],
            index=df.index
        )

        df = pd.concat(
            [df, poly_df],
            axis=1
        )

    num_df = df.select_dtypes(
        include=np.number
    )

    corr_matrix = num_df.corr().abs()

    upper = corr_matrix.where(
        np.triu(
            np.ones(
                corr_matrix.shape
            ),
            k=1
        ).astype(bool)
    )

    high_corr = [

        col

        for col in upper.columns

        if any(
            upper[col] > 0.95
        )
    ]

    if target in high_corr:
        high_corr.remove(target)

    if high_corr:

        df = df.drop(
            columns=high_corr
        )

    return {

        "processed_df":
            df,

        "retry_count":
            state["retry_count"] + 1,

        "feature_scores":
            feature_scores
    }

# ---------------------------------------------------------------------------
# Critic  – FIX: compare tuned score against pre-tuning best_score correctly
# ---------------------------------------------------------------------------

def critic_node(state):
    feedback   = []
    best_score = state["global_best_score"]          # FIX: use global best, not stale best_score
    tuned_score = state["tuning_results"].get("tuned_score", best_score)
    improvement = tuned_score - best_score

    if best_score < 0.60:
        feedback.append("Model performance is poor.")
        feedback.append("Feature engineering may be required.")
    elif best_score < 0.75:
        feedback.append("Model performance is moderate.")
        feedback.append("Further optimization may help.")
    else:
        feedback.append("Model performance is strong.")

    if improvement > 0.02:
        feedback.append("Hyperparameter tuning significantly improved performance.")
    elif improvement > 0:
        feedback.append("Hyperparameter tuning provided marginal gains.")
    else:
        feedback.append("Hyperparameter tuning did not improve performance.")

    if state["profile"]["dataset_size"] == "small":
        feedback.append("Small datasets may increase overfitting risk.")

    best_model = state["best_model"]
    if "random_forest" in best_model:
        feedback.append("Random Forest handled nonlinear relationships effectively.")
    elif "xgboost" in best_model:
        feedback.append("XGBoost captured complex feature interactions.")

    ensemble_score = state.get("ensemble_score", 0.0)
    if ensemble_score > best_score:
        feedback.append(
            f"Ensemble improved over best individual model "
            f"({ensemble_score:.4f} vs {best_score:.4f})."
        )
    elif ensemble_score > 0:
        feedback.append(
            f"Ensemble did not beat best individual model "
            f"({ensemble_score:.4f} vs {best_score:.4f})."
        )

    return {"critic_feedback": feedback}


# ---------------------------------------------------------------------------
# Conditional edges
# ---------------------------------------------------------------------------

def should_tune(state):
    """Only tune on the first pass. Retries skip straight to critic."""
    if state.get("retry_count", 0) > 0:
        return "end"
    if state["best_score"] < 0.70:
        return "tune"
    return "end"


def should_engineer_features(state):
    """Allow at most 1 feature-engineering retry."""
    if state["retry_count"] >= 1:
        return "end"
    if state["tuning_results"].get("tuned_score", 0) < 0.70:
        return "engineer"
    return "end"


# ---------------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------------

def data_quality_node(state):
    agent  = DataQualityAgent(state["processed_df"], state["target_column"])
    report = agent.analyze()
    return {"data_quality_report": report}


# ---------------------------------------------------------------------------
# LLM critic
# ---------------------------------------------------------------------------

def llm_critic_node(state):
    agent    = LLMCriticAgent()
    feedback = agent.analyze(state)
    return {"llm_feedback": feedback}


# ---------------------------------------------------------------------------
# Decision  – picks best individual model (never "ensemble")
# ---------------------------------------------------------------------------

def decision_node(state):
    history         = state["experiment_history"]
    best_experiment = max(history, key=lambda x: x["score"])

    reasoning = list(state.get("reasoning", []))
    if best_experiment["score"] > state["best_score"]:
        reasoning.append(
            f"Decision Agent restored best experiment: "
            f"{best_experiment['model']} ({best_experiment['score']:.4f})."
        )

    return {
        "best_model":       best_experiment["model"],
        "best_score":       best_experiment["score"],
        "global_best_model": best_experiment["model"],
        "global_best_score": best_experiment["score"],
        "reasoning":        reasoning,
    }


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def data_preparation_node(state):
    import pandas as pd
    from sklearn.preprocessing import OrdinalEncoder

    df     = state["processed_df"].copy()
    target = state["target_column"]

    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if col == target:
            continue
        df[col] = df[col].fillna(df[col].median())
        q1, q3  = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr     = q3 - q1
        df[col] = df[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    cat_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns if c != target]
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        df[cat_cols] = enc.fit_transform(df[cat_cols])

    return {"processed_df": df}


def explainability_node(state):

    best_model_name = state["best_model"]

    model = state["trained_models"][
        best_model_name
    ]

    df = state["processed_df"]

    target = state["target_column"]

    X = df.drop(
        columns=[target]
    )

    agent = ExplainabilityAgent(
        model,
        X
    )

    result = agent.explain()

    return result