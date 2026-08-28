"""Demo dashboard for the baseline agent (src/agent.py).

Run with: uv run python -m streamlit run src/ui/app.py
(python -m, not the bare `streamlit` binary - matches Research-agent's
convention, see that project's src/ui/app.py for why.)

Demo-scoped (2026-08-28): shows one agent's run, past and new. Not the
planner/modeller/critic loop the proposal describes.
"""

import pandas as pd
import streamlit as st

from src.agent import run_baseline
from src.report import RunReport, list_runs

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
.st-key-run_button button {
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


def _run_label(report: RunReport) -> str:
    return f"{report.timestamp[:19].replace('T', ' ')}  ·  acc {report.holdout_accuracy:.3f}"


def _render_report(report: RunReport) -> None:
    with st.container(horizontal=True):
        st.metric("Holdout accuracy", f"{report.holdout_accuracy:.1%}", border=True)
        st.metric("Train rows", report.train_rows, border=True)
        st.metric("Holdout rows", report.holdout_rows, border=True)
        st.metric("Run time", f"{report.duration_seconds:.1f}s", border=True)

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
            st.markdown(f"**Run ID**  \n`{report.run_id}`")
            st.caption("Scored against a fixed holdout the agent never saw - see ADR-005.")
            st.markdown("**Agent summary**")
            st.write(report.agent_summary)

    st.subheader("Generated training script", anchor=False)
    st.caption("The exact script the agent wrote and ran in the code-execution sandbox.")
    st.code(report.generated_code, language="python", line_numbers=True)

    with st.expander("Sandbox stdout", icon=":material/terminal:"):
        st.code(report.stdout or "(no output)", language="text")


st.html("""
<div class="hero-banner">
  <h1>Baseline critic</h1>
  <p>One agent trains a LightGBM baseline on Breast Cancer Wisconsin and is scored against a held-out split it never sees.</p>
</div>
""")

runs = list_runs()

with st.sidebar:
    st.subheader("Runs")
    if st.button("Run baseline", icon=":material/play_arrow:", type="primary", key="run_button"):
        try:
            with st.spinner("Agent is writing and training a baseline...", show_time=True):
                new_report = run_baseline()
            st.session_state["selected_run_id"] = new_report.run_id
            st.toast("Run complete", icon=":material/check_circle:")
            st.rerun()
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
            st.error(f"Run failed: {exc}", icon=":material/error:")

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

if not runs:
    st.info(
        "No runs yet. Click **Run baseline** in the sidebar to train the first one.",
        icon=":material/info:",
    )
else:
    selected_id = st.session_state.get("selected_run_id", runs[0].run_id)
    selected_report = next((r for r in runs if r.run_id == selected_id), runs[0])
    _render_report(selected_report)
