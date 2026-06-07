from langgraph.graph import StateGraph, END
from graph.state import AutoMLState
from graph.nodes import (
    profiling_node,
    task_node,
    training_node,
    evaluation_node,
    tuning_node,
    critic_node,
    feature_engineering_node,
    should_tune,
    should_engineer_features,
    data_quality_node,
    llm_critic_node,
    experiment_tracker_node,
    decision_node,
    data_preparation_node,
    model_selection_node,
    ensemble_node,
    ensemble_evaluation_node,
    explainability_node
)

graph = StateGraph(AutoMLState)

graph.add_node("profiling",profiling_node)
graph.add_node("task_detection",task_node)
graph.add_node("data_quality",data_quality_node)
graph.add_node("data_preparation",data_preparation_node)
graph.add_node("model_selection",model_selection_node)
graph.add_node("training",training_node)
graph.add_node("evaluation",evaluation_node)
graph.add_node("experiment_tracker",experiment_tracker_node)   # tracks individual models
graph.add_node("ensemble",ensemble_node)
graph.add_node("ensemble_evaluation",ensemble_evaluation_node)
graph.add_node("tuning", tuning_node)
graph.add_node("feature_engineering",feature_engineering_node)
graph.add_node("critic", critic_node)
graph.add_node("decision",             decision_node)
graph.add_node("llm_critic",           llm_critic_node)
graph.add_node("explainability", explainability_node)

graph.set_entry_point("profiling")

# Core pipeline
graph.add_edge("profiling",        "task_detection")
graph.add_edge("task_detection",   "data_quality")
graph.add_edge("data_quality",     "data_preparation")
graph.add_edge("data_preparation", "model_selection")
graph.add_edge("model_selection",  "training")
graph.add_edge("training",         "evaluation")

# FIX: track individual model results BEFORE ensemble, so experiment_history
# only contains individual models and decision_node picks a valid model name
graph.add_edge("evaluation","explainability")
graph.add_edge("explainability","experiment_tracker")

# Ensemble runs after individual tracking
graph.add_edge("experiment_tracker",  "ensemble")
graph.add_edge("ensemble",            "ensemble_evaluation")

# FIX: tune only on first pass; skip on retry passes
graph.add_conditional_edges(
    "ensemble_evaluation",
    should_tune,
    {
        "tune": "tuning",
        "end":  "critic",
    }
)

# FIX: optionally re-engineer once after tuning
graph.add_conditional_edges(
    "tuning",
    should_engineer_features,
    {
        "engineer": "feature_engineering",
        "end":      "critic",
    }
)

# After feature engineering, retrain - should_tune returns "end" because retry_count > 0
graph.add_edge("feature_engineering", "training")

# FIX: restore llm_critic before END (was missing in ChatGPT version)
graph.add_edge("critic",    "decision")
graph.add_edge("decision",  "llm_critic")
graph.add_edge("llm_critic", END)

workflow = graph.compile()