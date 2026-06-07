from graph.workflow import workflow
from graph.state import AutoMLState

initial_state: AutoMLState = {
    "dataset_path":              "datasets/sample.csv",
    "profile":                   {},
    "processed_df":              None,
    "target_column":             "",
    "task_type":                 "",
    "selected_models":           [],
    "model_selection_reasoning": [],
    "feature_names":             [],
    "training_results":          {},
    "trained_models":            {},
    "detailed_metrics":          {},
    "best_model":                "",
    "best_score":                0.0,
    "feature_importance":        {},
    "feature_scores":            {},
    "top_features":              [],
    "experiment_history":        [],
    "global_best_score":         0.0,
    "global_best_model":         "",
    "ensemble_model":            None,
    "ensemble_score":            0.0,
    "tuning_results":            {},
    "retry_count":               0,
    "reasoning":                 [],
    "critic_feedback":           [],
    "data_quality_report":       [],
    "llm_feedback":              "",
}

print("\n" + "=" * 60)
print("AGENTIC AUTOML WORKFLOW STARTED")
print("=" * 60)

result = workflow.invoke(initial_state)

print("\n" + "=" * 60)
print("WORKFLOW COMPLETED")
print("=" * 60)

print("\nDATASET PROFILE")
print("-" * 30)
p = result["profile"]
print(f"  Rows:         {p['rows']}")
print(f"  Columns:      {p['columns']}")
print(f"  Dataset Size: {p['dataset_size']}")

print("\nTASK INFORMATION")
print("-" * 30)
print(f"  Task Type:     {result['task_type']}")
print(f"  Target Column: {result['target_column']}")

print("\nMODEL SELECTION")
print("-" * 30)
for item in result["model_selection_reasoning"]:
    print(f"  - {item}")

print("\nMODEL RESULTS")
print("-" * 30)
for model, score in result["training_results"].items():
    print(f"  {model}: {score:.4f}")

print("\nBEST INDIVIDUAL MODEL")
print("-" * 30)
print(f"  Model: {result['best_model']}")
print(f"  Score: {result['best_score']:.4f}")

if result.get("ensemble_score", 0) > 0:
    print("\nENSEMBLE RESULTS")
    print("-" * 30)
    print(f"  Ensemble CV Score: {result['ensemble_score']:.4f}")
    winner = (
        "Ensemble"
        if result["ensemble_score"] > result["best_score"]
        else result["best_model"]
    )
    print(f"  Overall Winner:    {winner}")

if result["tuning_results"]:
    print("\nTUNING RESULTS")
    print("-" * 30)
    print("  Best Parameters:")
    for k, v in result["tuning_results"]["best_params"].items():
        print(f"    {k}: {v}")
    print(f"  Tuned Score: {result['tuning_results']['tuned_score']:.4f}")

print("\nTOP FEATURES (SHAP)")
print("-" * 30)
for feature, score in result.get("top_features", []):
    print(f"  {feature}: {score:.4f}")

print("\nDETAILED METRICS")
print("-" * 30)
for model, metrics in result["detailed_metrics"].items():
    print(f"\n  {model}")
    for key, value in metrics.items():
        if key != "confusion_matrix":
            print(f"    {key}: {value}")

print("\nEXPERIMENT HISTORY")
print("-" * 30)
for exp in result["experiment_history"]:
    print(f"  Iteration {exp['iteration']} | {exp['model']} | {exp['score']:.4f}")

print("\nGLOBAL BEST MODEL")
print("-" * 30)
print(f"  {result['global_best_model']}  ({result['global_best_score']:.4f})")

print("\nREASONING")
print("-" * 30)
for item in result["reasoning"]:
    print(f"  - {item}")

print("\nCRITIC FEEDBACK")
print("-" * 30)
for item in result["critic_feedback"]:
    print(f"  - {item}")

if result.get("llm_feedback"):
    print("\nLLM INSIGHTS")
    print("-" * 30)
    print(result["llm_feedback"])

print("\nWORKFLOW SUMMARY")
print("-" * 30)
print(f"  Feature Engineering Attempts: {result['retry_count']}")
print(f"  Models Evaluated:             {len(result['selected_models'])}")

print("\n" + "=" * 60)
print("END OF REPORT")
print("=" * 60)