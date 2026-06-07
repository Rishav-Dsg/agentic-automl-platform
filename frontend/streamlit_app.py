import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from agents.profiler_agent import ProfilerAgent
from agents.task_agent import TaskAgent
from agents.routing_agent import RoutingAgent
from agents.training_agent import TrainingAgent
from agents.evaluation_agent import EvaluationAgent
from agents.tuning_agent import TuningAgent
from agents.llm_critic_agent import LLMCriticAgent

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="Agentic AutoML", layout="wide")
st.title("🤖 Agentic AutoML Platform")

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    temp_path = "temp_dataset.csv"
    df.to_csv(temp_path, index=False)

    profiler = ProfilerAgent(temp_path)
    profile  = profiler.profile_dataset()

    st.subheader("Dataset Profile")
    st.json(profile)

    task_agent       = TaskAgent(df)
    possible_targets = task_agent.detect_target_column()
    target           = st.selectbox("Select Target Column", possible_targets)
    task_type        = task_agent.detect_task_type(target)

    st.write(f"**Detected Task:** `{task_type}`")

    routing_agent  = RoutingAgent(profile, task_type)
    selected_models = routing_agent.select_models()

    st.subheader("Selected Models")
    st.write(selected_models)

    if st.button("Run AutoML Pipeline"):

        # ----------------------------------------------------------------
        # Training
        # ----------------------------------------------------------------
        training_agent = TrainingAgent(df, target, selected_models, task_type)
        results, trained_models, detailed_metrics, feature_names = training_agent.train_models()

        st.subheader("Model Results")
        metric_label = "CV Accuracy" if task_type == "classification" else "CV R²"
        results_df = pd.DataFrame({
            "Model":       list(results.keys()),
            metric_label:  list(results.values())
        })
        st.dataframe(results_df)

        st.subheader("Model Comparison Chart")
        fig, ax = plt.subplots()
        ax.bar(results_df["Model"], results_df[metric_label])
        ax.set_ylabel(metric_label)
        ax.set_title("Model Performance")
        plt.xticks(rotation=15)
        st.pyplot(fig)

        # ----------------------------------------------------------------
        # Evaluation
        # ----------------------------------------------------------------
        evaluation_agent       = EvaluationAgent(results)
        best_model, best_score = evaluation_agent.select_best_model()

        st.subheader("Best Model")
        st.success(f"{best_model}  ({best_score:.4f})")

        reasoning = evaluation_agent.generate_reasoning(best_model, profile, task_type)
        st.subheader("Reasoning")
        for reason in reasoning:
            st.write(f"- {reason}")

        # ----------------------------------------------------------------
        # Hyperparameter tuning  –  FIX: tune actual best model
        # ----------------------------------------------------------------
        st.subheader("Hyperparameter Tuning")

        X_train, X_test, y_train, y_test = training_agent.prepare_data()

        tuning_agent = TuningAgent(X_train, y_train, best_model, task_type)

        with st.spinner(f"Tuning {best_model} with Optuna (20 trials)…"):
            best_params, tuned_score = tuning_agent.tune_best_model()

        st.write("**Best Parameters:**")
        st.json(best_params)
        st.write(f"**Tuned Score:** `{tuned_score:.4f}`")

        st.subheader("Tuning Improvement")
        comparison_df = pd.DataFrame({
            "Stage":        ["Initial", "Tuned"],
            metric_label:   [best_score, tuned_score]
        })
        fig2, ax2 = plt.subplots()
        ax2.bar(comparison_df["Stage"], comparison_df[metric_label])
        ax2.set_ylabel(metric_label)
        ax2.set_title("Before vs After Tuning")
        st.pyplot(fig2)

        # ----------------------------------------------------------------
        # Feature importance & SHAP (tree models only)
        # ----------------------------------------------------------------
        best_model_instance = trained_models[best_model]
        feature_names       = X_train.columns

        if "random_forest" in best_model or "xgboost" in best_model:
            st.subheader("Feature Importance")
            importances   = best_model_instance.feature_importances_
            importance_df = pd.DataFrame({
                "Feature":    feature_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False)

            st.dataframe(importance_df)

            fig3, ax3 = plt.subplots()
            ax3.barh(importance_df["Feature"], importance_df["Importance"])
            ax3.set_title("Feature Importance")
            ax3.set_xlabel("Importance Score")
            plt.gca().invert_yaxis()
            st.pyplot(fig3)

            st.subheader("SHAP Explainability")
            try:
                explainer   = shap.TreeExplainer(best_model_instance)
                shap_values = explainer.shap_values(X_train)
                plt.figure()
                shap.summary_plot(shap_values, X_train, show=False)
                st.pyplot(plt.gcf())
                plt.clf()
            except Exception as e:
                st.warning(f"SHAP plot skipped: {e}")
        else:
            importance_df = pd.DataFrame(columns=["Feature", "Importance"])

        # ----------------------------------------------------------------
        # Advanced metrics
        # ----------------------------------------------------------------
        st.subheader("Advanced Metrics")

        if task_type == "classification":
            metrics_df = pd.DataFrame([
                {
                    "Model":     model,
                    "Accuracy":  m["accuracy"],
                    "Precision": m["precision"],
                    "Recall":    m["recall"],
                    "F1 Score":  m["f1_score"]
                }
                for model, m in detailed_metrics.items()
            ])
            st.dataframe(metrics_df)

            st.subheader("Confusion Matrix")
            cm = detailed_metrics[best_model]["confusion_matrix"]
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
            ax_cm.set_title("Confusion Matrix")
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            st.pyplot(fig_cm)

        else:  # regression
            metrics_df = pd.DataFrame([
                {
                    "Model": model,
                    "R²":    m["r2"],
                    "MAE":   m["mae"],
                    "MSE":   m["mse"],
                    "RMSE":  m["rmse"]
                }
                for model, m in detailed_metrics.items()
            ])
            st.dataframe(metrics_df)

        # ----------------------------------------------------------------
        # PDF report
        # ----------------------------------------------------------------
        st.subheader("Download Report")
        report_path = "automl_report.pdf"
        doc         = SimpleDocTemplate(report_path)
        styles      = getSampleStyleSheet()
        elements    = []

        elements.append(Paragraph("Agentic AutoML Report", styles["Title"]))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f"Best Model: {best_model}",           styles["BodyText"]))
        elements.append(Paragraph(f"Best Score: {best_score:.4f}",       styles["BodyText"]))
        elements.append(Paragraph(f"Tuned Score: {tuned_score:.4f}",     styles["BodyText"]))
        elements.append(Paragraph(f"Task Type: {task_type}",             styles["BodyText"]))
        elements.append(Spacer(1, 12))

        for reason in reasoning:
            elements.append(Paragraph(f"• {reason}", styles["BodyText"]))

        doc.build(elements)

        with open(report_path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()

        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name="automl_report.pdf",
            mime="application/pdf"
        )

        # ----------------------------------------------------------------
        # LLM insights  –  FIX: pass real state
        # ----------------------------------------------------------------
        st.subheader("LLM Insights")

        top_features = []
        if not importance_df.empty:
            top_features = importance_df.head(5)["Feature"].tolist()

        # Build a minimal state dict for the LLM critic
        mock_state = {
            "profile":            profile,
            "task_type":          task_type,
            "best_model":         best_model,
            "best_score":         best_score,
            "tuning_results":     {"tuned_score": tuned_score, "best_params": best_params},
            "training_results":   results,
            "critic_feedback":    [],
            "data_quality_report":[],
            "retry_count":        0,
        }

        with st.spinner("Generating LLM insights…"):
            try:
                llm_agent = LLMCriticAgent()
                response  = llm_agent.analyze(mock_state)
                st.write(response)
            except Exception as e:
                st.warning(f"LLM insights unavailable: {e}")