"""CLI entry point for the Phase 5 evaluation harness (src/evaluation.py).

Builds each defect fixture once (cached under data/processed/eval_fixtures/,
--rebuild forces a fresh sandbox run), scores it against the critic
--trials times, and writes a summary JSON under
data/processed/eval_reports/ - the same file the Streamlit "Evaluation"
tab reads (src/evaluation.py's load_latest_summary()).

    uv run python -m scripts.run_evaluation
    uv run python -m scripts.run_evaluation --trials 10 --rebuild
"""

import argparse

from src.evaluation import DEFAULT_TRIALS_PER_FIXTURE, run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS_PER_FIXTURE, help="Critic trials per fixture.")
    parser.add_argument(
        "--rebuild", action="store_true", help="Re-run every fixture's sandbox script instead of using the cache."
    )
    args = parser.parse_args()

    summary = run_evaluation(trials_per_fixture=args.trials, rebuild=args.rebuild)

    print(f"\nModel: {summary['model']}")
    print(f"Overall detection rate: {summary['overall_detection_rate']:.2f}")
    print(f"Overall false-alarm rate: {summary['overall_false_alarm_rate']:.2f}")
    print(f"Tokens: {summary['total_prompt_tokens']} prompt, {summary['total_completion_tokens']} completion\n")

    header = f"{'category':<28}{'truth':<8}{'static':<8}{'llm-only':<10}{'combined':<10}{'accuracy':<10}"
    print(header)
    print("-" * len(header))
    for row in summary["categories"]:
        static_hit = "yes" if row["static_findings"] else "no"
        llm_only = f"{row['llm_only_reject_rate']:.2f}" if row["llm_only_reject_rate"] is not None else "n/a"
        print(
            f"{row['defect_category']:<28}{row['ground_truth_verdict']:<8}{static_hit:<8}"
            f"{llm_only:<10}{row['combined_reject_rate']:<10.2f}{row['holdout_accuracy']:<10.4f}"
        )


if __name__ == "__main__":
    main()
