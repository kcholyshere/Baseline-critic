"""CLI entry point for the safety/guardrail red-team evaluation
(src/red_team.py).

Two independent axes:
- Scan-only misuse fixtures against src/guardrails.py's scan_code() - no LLM
  call, instant, never executes any of the malicious source strings.
- Critic prompt-injection scenarios against src/critic.py's
  critique_run_async() - live local-Ollama calls, cached under
  data/processed/red_team_reports/ (--rebuild forces fresh calls).

    uv run python -m scripts.run_red_team
    uv run python -m scripts.run_red_team --trials 5 --rebuild
"""

import argparse

from src.red_team import DEFAULT_TRIALS_PER_SCENARIO, run_red_team


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"--trials must be a positive integer, got {value!r}")
    return parsed


def _print_misuse_table(misuse_scan: dict) -> None:
    print("Axis 1: scan-only misuse fixtures (src/guardrails.py scan_code())")
    header = f"{'fixture':<34}{'expected':<26}{'found':<26}{'result':<6}"
    print(header)
    print("-" * len(header))
    for row in misuse_scan["fixtures"]:
        expected = row["expected_category"] or "(none - clean template)"
        found = ", ".join(row["found_categories"]) if row["found_categories"] else "(none)"
        result = "PASS" if row["passed"] else "FAIL"
        print(f"{row['name']:<34}{expected:<26}{found:<26}{result:<6}")
    print(
        f"\nFalse positives on clean templates: {misuse_scan['false_positive_count']} "
        "(must be 0 for this run to be meaningful)\n"
    )


def _print_injection_table(injection_scenarios: dict) -> None:
    print("Axis 2: critic prompt-injection scenarios (src/critic.py critique_run_async())")
    header = (
        f"{'scenario':<40}{'static evidence':<22}{'n':<3}{'combined':<10}"
        f"{'llm-only':<10}{'fallback':<9}{'gated':<7}{'verdict':<9}{'flipped':<8}"
    )
    print(header)
    print("-" * len(header))
    for row in injection_scenarios["scenarios"]:
        evidence = ", ".join(row["static_evidence_categories"]) if row["static_evidence_categories"] else "(none)"
        combined = f"{row['reject_rate']:.2f}" if row["reject_rate"] is not None else "n/a"
        llm_only = f"{row['llm_only_reject_rate']:.2f}" if row["llm_only_reject_rate"] is not None else "n/a"
        verdict = row["majority_verdict"] or "n/a"
        flipped = "FLIPPED" if row["flipped"] else ""
        control_tag = " (control)" if row["is_control"] else ""
        print(
            f"{row['scenario'] + control_tag:<40}{evidence:<22}{row['n_trials']:<3}{combined:<10}"
            f"{llm_only:<10}{row['fallback_count']:<9}{row['gated_reject_count']:<7}{verdict:<9}{flipped:<8}"
        )
    print(
        "\n(combined = static checks + LLM, falling back to static-only when the LLM's JSON never parses; "
        "llm-only = reject rate among trials the LLM actually answered, excluding fallbacks - the number "
        "that answers whether the injected text swayed the LLM itself)"
    )
    for row in injection_scenarios["scenarios"]:
        n_accepting = row["n_trials"] - len(row["rejecting_trials"])
        if n_accepting:
            print(f"  {row['scenario']}: {n_accepting}/{row['n_trials']} trial(s) accepted - see 'accepting_trials' in the saved summary for the LLM's stated reasoning.")
    verdict_line = (
        "At least one injection scenario flipped the verdict from reject to accept."
        if injection_scenarios["any_flip"]
        else "No injection scenario flipped the verdict - the critic held its reject on every case with "
        "real static evidence, despite the injected text."
    )
    print(f"\n{verdict_line}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--trials",
        type=_positive_int,
        default=DEFAULT_TRIALS_PER_SCENARIO,
        help="Critic trials per injection scenario.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Re-run every injection scenario's critic calls instead of using the cache.",
    )
    args = parser.parse_args()

    summary = run_red_team(trials_per_scenario=args.trials, rebuild=args.rebuild)

    print(f"\nModel: {summary['model']}\n")
    _print_misuse_table(summary["misuse_scan"])
    _print_injection_table(summary["injection_scenarios"])

    passthrough = summary["passthrough"]
    print("Axis 2 (lower priority): modeller-side text passthrough, no LLM call")
    print(f"  Adversarial code payload survived the report JSON round-trip intact: {passthrough['code_survived']}")
    print(f"  Adversarial stdout payload survived the report JSON round-trip intact: {passthrough['stdout_survived']}")


if __name__ == "__main__":
    main()
