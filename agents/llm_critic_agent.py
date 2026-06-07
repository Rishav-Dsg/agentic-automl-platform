from langchain_ollama import ChatOllama

class LLMCriticAgent:

    def __init__(self):
        self.llm = ChatOllama(model="qwen2.5", temperature=0)

    def analyze(self, state: dict) -> str:
        """
        FIX: Build a rich, state-aware prompt so the LLM produces
        genuinely useful analysis instead of a hardcoded stub reply.
        """
        profile      = state.get("profile", {})
        task_type    = state.get("task_type", "unknown")
        best_model   = state.get("best_model", "unknown")
        best_score   = state.get("best_score", 0.0)
        tuning_results = state.get("tuning_results", {})
        tuned_score  = tuning_results.get("tuned_score", best_score)
        training_results = state.get("training_results", {})
        critic_feedback  = state.get("critic_feedback", [])
        data_quality_report = state.get("data_quality_report", [])
        retry_count  = state.get("retry_count", 0)

        # Build model scores string
        model_scores_str = "\n".join(
            f"  - {m}: {s:.4f}" for m, s in training_results.items()
        )

        # Build critic feedback string
        critic_str = "\n".join(f"  - {f}" for f in critic_feedback)
        quality_str = "\n".join(f"  - {f}" for f in data_quality_report)

        prompt = f"""You are a senior ML engineer reviewing an AutoML pipeline run. Provide a concise but insightful analysis.

DATASET SUMMARY:
- Rows: {profile.get('rows', 'N/A')}
- Columns: {profile.get('columns', 'N/A')}
- Dataset size category: {profile.get('dataset_size', 'N/A')}
- Sparsity: {profile.get('sparsity', 0):.2%}
- Numerical columns: {len(profile.get('numerical_columns', []))}
- Categorical columns: {len(profile.get('categorical_columns', []))}

TASK: {task_type}

DATA QUALITY FINDINGS:
{quality_str if quality_str else '  None reported'}

MODEL COMPARISON (cross-validated scores):
{model_scores_str if model_scores_str else '  No results'}

BEST MODEL: {best_model}
  - Initial CV score:  {best_score:.4f}
  - After tuning:      {tuned_score:.4f}
  - Improvement:       {tuned_score - best_score:+.4f}

FEATURE ENGINEERING RETRIES: {retry_count}

RULE-BASED CRITIC FEEDBACK:
{critic_str if critic_str else '  None'}

Please address the following:
1. Why did {best_model} likely outperform the other models given this dataset profile?
2. Why might the other models have underperformed?
3. What dataset characteristics (size, sparsity, column mix) most influenced results?
4. Is the current performance level acceptable for a production system? What threshold would you recommend for this task?
5. Top 3 concrete next steps to further improve performance.
6. Overfitting risk assessment given the dataset size and tuned vs CV score gap.

Be specific and dataset-aware. Avoid generic advice.
"""

        response = self.llm.invoke(prompt)

        # response.content is typed as str | list[str | dict] by LangChain.
        # Normalise to a plain string so our return type is always str.
        content = response.content
        if isinstance(content, str):
            return content
        # If it's a list of content blocks, extract the text from each block.
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", str(block)))
        return "\n".join(parts)