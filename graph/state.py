from typing import TypedDict, Any

class AutoMLState(TypedDict):
    # Input
    dataset_path: str

    # Profiling
    profile: dict
    processed_df: Any

    # Task
    target_column: str
    task_type: str

    # Model selection
    selected_models: list
    model_selection_reasoning: list
    feature_names: list

    # Training & evaluation
    training_results: dict
    trained_models: dict
    detailed_metrics: dict
    best_model: str
    best_score: float

    # Explainability
    feature_importance: dict   # {feature_name: float}
    top_features: list         # [(feature_name, score), ...]
    feature_scores: dict
    
    # Experiment tracking
    experiment_history: list
    global_best_score: float
    global_best_model: str

    # Ensemble
    ensemble_model: Any
    ensemble_score: float

    # Tuning
    tuning_results: dict

    # Feature engineering retry counter
    retry_count: int

    # Feedback & reasoning
    reasoning: list
    critic_feedback: list
    data_quality_report: list
    llm_feedback: str