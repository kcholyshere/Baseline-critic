"""Demo dashboard for the baseline agent (src/agent.py).

Run with: uv run python -m streamlit run src/ui/app.py
(python -m, not the bare `streamlit` binary - matches Research-agent's
convention, see that project's src/ui/app.py for why.)

Demo-scoped (2026-08-28): shows one agent's run, past and new, against
either the built-in Breast Cancer Wisconsin dataset or an uploaded CSV.
Not the planner/modeller/critic loop the proposal describes. Binary
classification only.
"""

import asyncio

import pandas as pd
import streamlit as st

from src import config, dataset, evaluation
from src.agent import rescore_run_async, run_baseline, run_baseline_for
from src.profiling import profile_dataframe
from src.report import RunReport, list_runs, save_rescoring

st.set_page_config(
    page_title="Baseline critic",
    page_icon=":material/analytics:",
    layout="wide",
)

st.html("""
<style>
.hero-banner {
    padding: 1.75rem 2rem;
    border-radius: 14px;
    background: linear-gradient(135deg, #2F6F5E 0%, #24564A 100%);
    color: #FBFAF7;
    margin-bottom: 1.5rem;
}
.hero-banner h1 {
    color: #FBFAF7;
    margin: 0 0 0.25rem 0;
    font-size: 1.9rem;
}
.hero-banner p {
    color: #DCEEE7;
    margin: 0;
    font-size: 0.95rem;
}
.st-key-run_button button, .st-key-run_button_upload button {
    width: 100%;
    font-weight: 600;
}
</style>
""")


def _classification_report_df(report: dict) -> pd.DataFrame:
    rows = {label: values for label, values in report.items() if isinstance(values, dict)}
    df = pd.DataFrame(rows).T.rename(columns={"f1-score": "f1_score"})
    df["support"] = df["support"].astype(int)
    return df[["precision", "recall", "f1_score", "support"]].round(3)


def _profile_df(profile: dict) -> pd.DataFrame:
    rows = []
    for name, stats in profile["features"].items():
        rows.append(
            {
                "column": name,
                "dtype": stats["dtype"],
                "cardinality": stats["cardinality"],
                "missing": f"{stats['missing_count']} ({stats['missing_pct']}%)",
                "zeros": stats["zero_count"] if stats["zero_count"] is not None else "-",
                "likely identifier": "yes" if stats["likely_identifier"] else "",
            }
        )
    return pd.DataFrame(rows).set_index("column")


def _run_label(report: RunReport) -> str:
    return f"{report.timestamp[:19].replace('T', ' ')}  ·  acc {report.holdout_accuracy:.3f}"


def _render_report(report: RunReport) -> None:
    with st.container(horizontal=True):
        st.metric("Holdout accuracy", f"{report.holdout_accuracy:.1%}", border=True)
        st.metric("Train rows", report.train_rows, border=True)
        st.metric("Holdout rows", report.holdout_rows, border=True)
        st.metric("Run time", f"{report.duration_seconds:.1f}s", border=True)

    critic_tokens = (report.critique or {}).get("prompt_tokens", 0) + (report.critique or {}).get(
        "completion_tokens", 0
    )
    modeller_tokens = report.modeller_prompt_tokens + report.modeller_completion_tokens
    with st.container(horizontal=True):
        st.metric("Modeller tokens", modeller_tokens, border=True)
        st.metric("Critic tokens", critic_tokens, border=True)
        st.metric("Training attempts", report.attempts, border=True)
        if report.critique is not None:
            st.metric("Critique rounds", report.critique["rounds_used"], border=True)

    if report.critique is not None:
        with st.container(horizontal=True, vertical_alignment="center"):
            if report.critique["verdict"] == "accept":
                st.badge("Critic: accepted", icon=":material/check_circle:", color="green")
            else:
                st.badge(
                    f"Critic: rejected - {report.critique['defect_category']}",
                    icon=":material/gpp_maybe:",
                    color="red",
                )
            if report.dataset_name == "breast_cancer_wisconsin":
                if st.button("Re-score", icon=":material/refresh:", key=f"rescore_{report.run_id}"):
                    with st.spinner("Re-running static checks and critic on the stored run...", show_time=True):
                        dataset.build_train_artifact()
                        new_critique = asyncio.run(rescore_run_async(report, dataset.TRAIN_PATH))
                        save_rescoring(report.run_id, new_critique.to_dict(), config.DEFAULT_MODEL_URI)
                    st.toast("Re-scored - see the panel below", icon=":material/check_circle:")
                    st.session_state[f"rescore_result_{report.run_id}"] = new_critique.to_dict()

        rescore_result = st.session_state.get(f"rescore_result_{report.run_id}")
        if rescore_result is not None:
            with st.expander("Latest re-score result", icon=":material/history:", expanded=True):
                st.caption(
                    "Computed just now from the stored code and results, without re-running the "
                    "modeller agent - the original verdict above is unchanged."
                )
                verdict_label = (
                    "accept" if rescore_result["verdict"] == "accept" else f"reject - {rescore_result['defect_category']}"
                )
                st.markdown(f"**Verdict:** {verdict_label}")
                if rescore_result["defect"]:
                    st.markdown(f"**Defect:** {rescore_result['defect']}")
                st.caption(rescore_result["evidence"])

    col_report, col_meta = st.columns([3, 2])
    with col_report:
        with st.container(border=True):
            st.subheader("Classification report")
            st.dataframe(
                _classification_report_df(report.classification_report),
                width="stretch",
                column_config={
                    "precision": st.column_config.NumberColumn(format="%.3f"),
                    "recall": st.column_config.NumberColumn(format="%.3f"),
                    "f1_score": st.column_config.NumberColumn("F1 score", format="%.3f"),
                    "support": st.column_config.NumberColumn(width="small"),
                },
            )
    with col_meta:
        with st.container(border=True):
            st.subheader("Run")
            st.markdown(f"**Dataset**  \n{report.dataset_name}")
            st.markdown(f"**Target column**  \n{report.target_column}")
            if report.positive_class:
                st.markdown(f"**Positive class**  \n{report.positive_class}")
            st.markdown(f"**Run ID**  \n`{report.run_id}`")
            st.caption("Scored against a fixed holdout the agent never saw - see ADR-005.")
            st.markdown("**Agent summary**")
            st.write(report.agent_summary)

    st.subheader("Generated training script", anchor=False)
    st.caption("The exact script the agent wrote and ran in the code-execution sandbox.")
    st.code(report.generated_code, language="python", line_numbers=True)

    if report.critique is not None and report.critique["verdict"] == "reject":
        with st.expander("Critic's defect", icon=":material/gpp_maybe:", expanded=True):
            st.markdown(f"**{report.critique['defect']}**")
            st.caption(report.critique["evidence"])
            if report.critique["static_findings"]:
                st.markdown("Deterministic static-check findings:")
                for finding in report.critique["static_findings"]:
                    st.markdown(f"- {finding}")

    with st.expander("Sandbox stdout", icon=":material/terminal:"):
        st.code(report.stdout or "(no output)", language="text")


def _render_evaluation_tab() -> None:
    st.caption(
        "Phase 5: plants one known defect per defect_category into the Breast Cancer Wisconsin "
        "dataset and measures the critic's detection rate against its false-alarm rate on a "
        "clean run - see ADR-010 and references/project-proposal.md."
    )
    if st.button("Run evaluation now", icon=":material/science:", key="run_evaluation_button"):
        with st.spinner(
            "Scoring the critic against every defect fixture - this takes several minutes...", show_time=True
        ):
            evaluation.run_evaluation()
        st.toast("Evaluation complete", icon=":material/check_circle:")
        st.rerun()

    summary = evaluation.load_latest_summary()
    if summary is None:
        st.info(
            "No evaluation run yet. Click **Run evaluation now**, or run "
            "`uv run python -m scripts.run_evaluation` from the command line.",
            icon=":material/info:",
        )
        return

    st.caption(
        f"Last run: {summary['generated_at'][:19].replace('T', ' ')}  ·  model: {summary['model']}  ·  "
        f"{summary['trials_per_fixture']} critic trials per category"
    )
    with st.container(horizontal=True):
        st.metric("Detection rate", f"{summary['overall_detection_rate']:.0%}", border=True)
        st.metric("False-alarm rate", f"{summary['overall_false_alarm_rate']:.0%}", border=True)
        st.metric("Prompt tokens", summary["total_prompt_tokens"], border=True)
        st.metric("Completion tokens", summary["total_completion_tokens"], border=True)

    rows = [
        {
            "category": row["defect_category"],
            "ground truth": row["ground_truth_verdict"],
            "static check fired": "yes" if row["static_findings"] else "no",
            "LLM-only reject rate": row["llm_only_reject_rate"],
            "combined reject rate": row["combined_reject_rate"],
            "holdout accuracy": row["holdout_accuracy"],
        }
        for row in summary["categories"]
    ]
    st.dataframe(
        pd.DataFrame(rows).set_index("category"),
        width="stretch",
        column_config={
            "LLM-only reject rate": st.column_config.NumberColumn(format="%.2f"),
            "combined reject rate": st.column_config.NumberColumn(format="%.2f"),
            "holdout accuracy": st.column_config.NumberColumn(format="%.4f"),
        },
    )
    st.caption(
        "target_leakage and temporal_leakage have no deterministic static check (src/checks.py) - "
        "their reject rate is the critic's own judgement, not the checks running alongside it."
    )

    with st.expander("Fixture descriptions"):
        for row in summary["categories"]:
            st.markdown(f"**{row['defect_category']}** ({row['ground_truth_verdict']}): {row['description']}")

    clean_row = next((r for r in summary["categories"] if r["ground_truth_verdict"] == "accept"), None)
    if clean_row and clean_row["rejecting_trials"]:
        with st.expander(
            f"Why the critic rejected the clean run ({len(clean_row['rejecting_trials'])}/"
            f"{clean_row['n_trials']} trials)",
            icon=":material/warning:",
        ):
            st.caption(
                "The false-alarm rate above is only useful once you know what the critic "
                "actually objected to on a run with no real defect."
            )
            for trial in clean_row["rejecting_trials"]:
                st.markdown(f"- **{trial['defect_category']}**: {trial['defect']}")


st.html("""
<div class="hero-banner">
  <h1>Baseline critic</h1>
  <p>One agent trains a LightGBM baseline and is scored against a held-out split it never sees - the built-in demo dataset, or your own CSV.</p>
</div>
""")

upload_df: pd.DataFrame | None = None
upload_file_name: str | None = None
upload_target_column: str | None = None
upload_positive_class: str | None = None
upload_negative_class: str | None = None
upload_profile: dict | None = None

with st.sidebar:
    st.subheader("Run a baseline")
    mode = st.radio(
        "Dataset",
        options=["Demo dataset", "Upload your own CSV"],
        key="dataset_mode",
        label_visibility="collapsed",
    )

    if mode == "Demo dataset":
        st.caption("Breast Cancer Wisconsin, target: diagnosis")
        if st.button("Run baseline", icon=":material/play_arrow:", type="primary", key="run_button"):
            try:
                with st.spinner("Agent is writing and training a baseline...", show_time=True):
                    new_report = run_baseline()
                st.session_state["selected_run_id"] = new_report.run_id
                st.toast("Run complete", icon=":material/check_circle:")
                st.rerun()
            except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
                st.error(f"Run failed: {exc}", icon=":material/error:")

    else:
        st.caption("Binary classification only for this version.")
        uploaded_file = st.file_uploader("CSV file", type="csv", key="upload_file")
        if uploaded_file is not None:
            upload_df = pd.read_csv(uploaded_file)
            upload_file_name = uploaded_file.name
            upload_target_column = st.selectbox(
                "Target column", options=upload_df.columns.tolist(), key="upload_target_column"
            )
            try:
                classes = list(dataset.validate_binary_target(upload_df, upload_target_column))
            except ValueError as exc:
                st.error(str(exc), icon=":material/error:")
            else:
                upload_positive_class = st.selectbox(
                    "Positive class (outcome of interest)",
                    options=classes,
                    key="upload_positive_class",
                )
                upload_negative_class = next(c for c in classes if c != upload_positive_class)
                upload_profile = profile_dataframe(upload_df, upload_target_column)

                if st.button(
                    "Run baseline", icon=":material/play_arrow:", type="primary", key="run_button_upload"
                ):
                    try:
                        with st.spinner("Agent is writing and training a baseline...", show_time=True):
                            uploaded = dataset.prepare_uploaded_dataset(
                                upload_df, upload_target_column, upload_file_name
                            )
                            new_report = run_baseline_for(
                                train_path=uploaded.train_path,
                                target_column=upload_target_column,
                                positive_class=upload_positive_class,
                                negative_class=upload_negative_class,
                                holdout=uploaded.holdout,
                                dataset_name=uploaded.dataset_name,
                            )
                        st.session_state["selected_run_id"] = new_report.run_id
                        st.toast("Run complete", icon=":material/check_circle:")
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
                        st.error(f"Run failed: {exc}", icon=":material/error:")

    st.divider()
    st.subheader("Runs")
    runs = list_runs()
    if runs:
        labels = {r.run_id: _run_label(r) for r in runs}
        default_id = st.session_state.get("selected_run_id", runs[0].run_id)
        default_index = next((i for i, r in enumerate(runs) if r.run_id == default_id), 0)
        selected_id = st.selectbox(
            "Past runs",
            options=[r.run_id for r in runs],
            format_func=lambda run_id: labels[run_id],
            index=default_index,
            label_visibility="collapsed",
        )
        st.session_state["selected_run_id"] = selected_id

tab_run, tab_evaluation = st.tabs(["Run", "Evaluation harness"])

with tab_run:
    if upload_profile is not None:
        with st.container(border=True):
            st.subheader(f"Dataset profile: {upload_file_name}", anchor=False)
            st.caption(
                f"{upload_profile['row_count']} rows, {len(upload_profile['features'])} feature columns. "
                f"Target '{upload_target_column}' balance: {upload_profile['target']['value_counts']}"
            )
            st.dataframe(_profile_df(upload_profile), width="stretch")
        st.divider()

    if not runs:
        st.info(
            "No runs yet. Click **Run baseline** in the sidebar to train the first one.",
            icon=":material/info:",
        )
    else:
        selected_id = st.session_state.get("selected_run_id", runs[0].run_id)
        selected_report = next((r for r in runs if r.run_id == selected_id), runs[0])
        _render_report(selected_report)

with tab_evaluation:
    _render_evaluation_tab()
