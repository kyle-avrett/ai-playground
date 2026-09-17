from collections import defaultdict

import gradio as gr
import pandas as pd
from eval import evaluate_all_answers, evaluate_all_retrieval

RETRIEVAL_THRESHOLDS = {
    "mrr": (0.9, 0.75),
    "ndcg": (0.9, 0.75),
    "coverage": (90.0, 75.0),
}
ANSWER_THRESHOLDS = {
    "accuracy": (4.5, 4.0),
    "completeness": (4.5, 4.0),
    "relevance": (4.5, 4.0),
}
PROGRESS = gr.Progress()


def get_color(value: float, metric_type: str) -> str:
    thresholds = RETRIEVAL_THRESHOLDS | ANSWER_THRESHOLDS
    if metric_type not in thresholds:
        return "black"

    green, amber = thresholds[metric_type]
    if value >= green:
        return "green"
    if value >= amber:
        return "orange"
    return "red"


def format_metric_html(
    label: str,
    value: float,
    metric_type: str,
    is_percentage: bool = False,
    score_format: bool = False,
) -> str:
    color = get_color(value, metric_type)
    if is_percentage:
        value_text = f"{value:.1f}%"
    elif score_format:
        value_text = f"{value:.2f}/5"
    else:
        value_text = f"{value:.4f}"

    return f"""
    <div style="margin: 10px 0; padding: 15px; background-color: #f5f5f5; border-radius: 8px; border-left: 5px solid {color};">
        <div style="font-size: 14px; color: #666; margin-bottom: 5px;">{label}</div>
        <div style="font-size: 28px; font-weight: bold; color: {color};">{value_text}</div>
    </div>
    """


def complete_html(count: int) -> str:
    return f"""
    <div style="margin-top: 20px; padding: 10px; background-color: #d4edda; border-radius: 5px; text-align: center; border: 1px solid #c3e6cb;">
        <span style="font-size: 14px; color: #155724; font-weight: bold;">✓ Evaluation Complete: {count} tests</span>
    </div>
    """


def category_frame(category_scores, value_column):
    return pd.DataFrame(
        {
            "Category": category,
            value_column: sum(scores) / len(scores),
        }
        for category, scores in category_scores.items()
    )


def run_retrieval_evaluation(progress=PROGRESS):
    total_mrr = 0.0
    total_ndcg = 0.0
    total_coverage = 0.0
    category_mrr = defaultdict(list)
    count = 0

    for test, result, progress_value in evaluate_all_retrieval():
        count += 1
        total_mrr += result.mrr
        total_ndcg += result.ndcg
        total_coverage += result.keyword_coverage
        category_mrr[test.category].append(result.mrr)
        progress(progress_value, desc=f"Evaluating test {count}...")

    final_html = f"""
    <div style="padding: 0;">
        {format_metric_html("Mean Reciprocal Rank (MRR)", total_mrr / count, "mrr")}
        {format_metric_html("Normalized DCG (nDCG)", total_ndcg / count, "ndcg")}
        {format_metric_html("Keyword Coverage", total_coverage / count, "coverage", is_percentage=True)}
        {complete_html(count)}
    </div>
    """

    return final_html, category_frame(category_mrr, "Average MRR")


def run_answer_evaluation(progress=PROGRESS):
    total_accuracy = 0.0
    total_completeness = 0.0
    total_relevance = 0.0
    category_accuracy = defaultdict(list)
    count = 0

    for test, result, progress_value in evaluate_all_answers():
        count += 1
        total_accuracy += result.accuracy
        total_completeness += result.completeness
        total_relevance += result.relevance
        category_accuracy[test.category].append(result.accuracy)
        progress(progress_value, desc=f"Evaluating test {count}...")

    final_html = f"""
    <div style="padding: 0;">
        {format_metric_html("Accuracy", total_accuracy / count, "accuracy", score_format=True)}
        {format_metric_html("Completeness", total_completeness / count, "completeness", score_format=True)}
        {format_metric_html("Relevance", total_relevance / count, "relevance", score_format=True)}
        {complete_html(count)}
    </div>
    """

    return final_html, category_frame(category_accuracy, "Average Accuracy")


def main():
    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="RAG Evaluation Dashboard", theme=theme) as app:
        gr.Markdown("# 📊 RAG Evaluation Dashboard")
        gr.Markdown(
            "Evaluate retrieval and answer quality for the Insurellm RAG system"
        )

        gr.Markdown("## 🔍 Retrieval Evaluation")
        retrieval_button = gr.Button("Run Evaluation", variant="primary", size="lg")
        with gr.Row():
            with gr.Column(scale=1):
                retrieval_metrics = gr.HTML(
                    "<div style='padding: 20px; text-align: center; color: #999;'>Click 'Run Evaluation' to start</div>"
                )
            with gr.Column(scale=1):
                retrieval_chart = gr.BarPlot(
                    x="Category",
                    y="Average MRR",
                    title="Average MRR by Category",
                    y_lim=[0, 1],
                    height=400,
                )

        gr.Markdown("## 💬 Answer Evaluation")
        answer_button = gr.Button("Run Evaluation", variant="primary", size="lg")
        with gr.Row():
            with gr.Column(scale=1):
                answer_metrics = gr.HTML(
                    "<div style='padding: 20px; text-align: center; color: #999;'>Click 'Run Evaluation' to start</div>"
                )
            with gr.Column(scale=1):
                answer_chart = gr.BarPlot(
                    x="Category",
                    y="Average Accuracy",
                    title="Average Accuracy by Category",
                    y_lim=[1, 5],
                    height=400,
                )

        retrieval_button.click(
            fn=run_retrieval_evaluation,
            outputs=[retrieval_metrics, retrieval_chart],
        )
        answer_button.click(
            fn=run_answer_evaluation,
            outputs=[answer_metrics, answer_chart],
        )

    app.launch(inbrowser=True)


if __name__ == "__main__":
    main()
