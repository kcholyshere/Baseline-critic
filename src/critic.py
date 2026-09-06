"""The critic agent (Phase 3): reviews a completed run and either accepts
the result or names one defect - the proposal's actual differentiator
("a score is not a result until the critic has failed to explain it away",
references/project-proposal.md).

Given deliberately different information from the modeller agent
(src/agent.py): the generated code, the results, the dataset profile, and
this module's own deterministic static-check findings (src/checks.py) - but
not the modeller's system instruction (so it judges what the code actually
does, not whether it "followed orders"), and not the holdout rows or a
code-execution tool (the modeller is already structurally blind to the
holdout per ADR-005; giving the critic either would reopen that boundary
for no benefit, since this phase never lets the critic retrain anything).

"Bounded so it cannot argue forever" (the proposal's own phrase) is
enforced in code, not just asked for in the prompt - the same lesson
ADR-008 already drew from the modeller agent: MAX_CRITIQUE_ROUNDS caps how
many times an unparseable response gets retried, and critique_run always
returns a Critique, falling back to a verdict implied by the static
findings alone rather than raising or hanging.

The reject-gate (ADR-016) closes the general critic hallucination class
identified during Phase 5 calibration (agent_docs/decisions.md ADR-013,
ADR-015): a reject naming a defect_category that no static check actually
found evidence for is treated the same as an unparseable response and
retried, rather than trusted outright. temporal_leakage is exempt by
design - ADR-014 found no safe static threshold for it, so it stays
critic-only, not silently caught by a gate with nothing behind it.
"""

import json
from dataclasses import asdict, dataclass
from typing import Literal

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from google.genai import types as genai_types
from pydantic import BaseModel

from src.checks import StaticFinding
from src.report import RunReport

APP_NAME = "baseline_critic_review"
USER_ID = "demo-user"
MAX_CRITIQUE_ROUNDS = 3

DefectCategory = Literal[
    "target_leakage",
    "train_test_contamination",
    "temporal_leakage",
    "unseeded_randomness",
    "degenerate_split",
    "score_mismatch",
    "none",
]

# temporal_leakage has no static check by design (ADR-014: no correlation
# threshold safely separates it from a legitimately strong feature) - it
# stays exempt from the reject-gate below rather than being blocked by a
# check that was never built.
UNGATED_CATEGORIES = {"temporal_leakage", "none"}


@dataclass
class Critique:
    verdict: Literal["accept", "reject"]
    defect_category: DefectCategory
    defect: str
    evidence: str
    static_findings: list[str]
    rounds_used: int
    fallback_used: bool = False
    gated_rejects: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class _CriticVerdict(BaseModel):
    """The LLM's own output, before static_findings/rounds_used are attached."""

    verdict: Literal["accept", "reject"]
    defect_category: DefectCategory
    defect: str
    evidence: str


def _build_critic_instruction(report: RunReport, profile_text: str, static_findings: list[StaticFinding]) -> str:
    findings_block = (
        "\n".join(f"- {finding.text}" for finding in static_findings)
        if static_findings
        else "(none - the deterministic checks found nothing)"
    )
    generated_code = report.generated_code
    stdout = report.stdout
    agent_summary = report.agent_summary
    if report.task_type == "regression":
        metrics = report.regression_metrics
        results_block = (
            f"Reported holdout R²: {metrics.get('r2', report.holdout_accuracy):.4f}\n"
            f"Reported holdout RMSE: {metrics.get('rmse', float('nan')):.4f}\n"
            f"Reported holdout MAE: {metrics.get('mae', float('nan')):.4f}"
        )
    else:
        classification_report_json = json.dumps(report.classification_report)
        results_block = (
            f"Reported holdout accuracy: {report.holdout_accuracy:.4f}\n"
            f"Classification report: {classification_report_json}"
        )
    return f"""\
You are the critic in a baseline modelling pipeline. A separate agent wrote
and ran a training script; your only job is to decide whether its reported
result is real, and to reject it with exactly one named defect if it is not.
Your success is measured by correctly rejecting bad results, not by being
agreeable.

You do not see the agent's own instructions or the held-out data it was
scored against - only what is below. Judge the code by what it actually
does, not by what it claims to have done.

Deterministic dataset profile:
{profile_text}

Deterministic static-check findings on the generated code (computed with
plain Python, not an LLM - trust these over your own reading of the code
where they disagree):
{findings_block}

The generated training script:
```python
{generated_code}
```

The script's own stdout:
{stdout}

The agent's one-sentence summary of what it did:
{agent_summary}

{results_block}

Look specifically for: leakage, train/test contamination, a target column
smuggled into the features, temporal leakage, unseeded randomness, and any
mismatch between what the code claims to measure and what it actually
measures. If you find a real defect, reject with exactly one
defect_category and one sentence naming it in "defect". If nothing above
gives you a concrete reason to doubt the result, accept - do not reject on
vague suspicion alone, since a dataset can legitimately be easy.

Respond with only a single JSON object, nothing else - no markdown code
fence, no commentary before or after it. It must have exactly these keys:
"verdict" ("accept" or "reject"), "defect_category" (one of
{list(DefectCategory.__args__)} - "none" if accepting), "defect" (one
sentence, empty string if accepting), "evidence" (what you looked at to
reach this verdict).
"""


def _build_critic_agent(instruction: str, model: str | LiteLlm) -> Agent:
    # No output_schema: grammar-constrained JSON decoding was measured to hang
    # (600s+ timeout) with this local model on a prompt this size/shape,
    # even though the exact same prompt without the constraint answers in
    # seconds. Free-form JSON, parsed leniently below, is what the modeller
    # agent's own instruction-following already relies on successfully.
    #
    # instruction is passed as a callable, not a plain string: ADK only
    # regex-templates a plain-string instruction, scanning it for bare
    # {identifier} patterns to substitute from session state (raising
    # KeyError if not found) - a callable ("InstructionProvider") bypasses
    # that scan entirely. The generated code/stdout embedded here can and
    # does contain a literal f-string placeholder like {val_accuracy},
    # which is indistinguishable from a real template variable to that scan.
    return Agent(
        name="critic_agent",
        model=model,
        instruction=lambda _ctx: instruction,
    )


_CATEGORY_ALIASES = {
    "train/test contamination": "train_test_contamination",
    "train-test contamination": "train_test_contamination",
    "target leakage": "target_leakage",
    "temporal leakage": "temporal_leakage",
    "unseeded randomness": "unseeded_randomness",
    "degenerate split": "degenerate_split",
    "score mismatch": "score_mismatch",
}


def _normalise_category(value: str) -> str:
    value = value.strip().lower()
    return _CATEGORY_ALIASES.get(value, value.replace(" ", "_").replace("-", "_"))


def _parse_verdict(text: str) -> _CriticVerdict | None:
    """Extracts and validates a JSON object from free-form model output -
    strips a markdown code fence if present, tolerates near-miss
    defect_category spelling (the model isn't grammar-constrained, so exact
    strings aren't guaranteed), returns None on any failure rather than
    raising, matching what critique_run_async expects to retry against.

    Also rejects a self-inconsistent verdict: "reject" paired with
    defect_category "none" (ADR-010/ADR-011 - the eval harness found the
    model producing exactly this combination, with an empty defect string,
    which pydantic's per-field validation can't catch since each field is
    individually valid). Treated the same as an unparseable response, so
    critique_run_async retries it rather than reporting a reject with no
    named defect."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("```")[1]
        stripped = stripped.removeprefix("json").strip()
    try:
        data = json.loads(stripped)
        if "defect_category" in data:
            data["defect_category"] = _normalise_category(str(data["defect_category"]))
        verdict = _CriticVerdict.model_validate(data)
        if verdict.verdict == "reject" and verdict.defect_category == "none":
            return None
        return verdict
    except Exception:
        return None


async def _run_critic_round(instruction: str, model: str | LiteLlm) -> tuple[_CriticVerdict | None, int, int]:
    """Runs one critic call and returns (parsed verdict or None, prompt_tokens,
    completion_tokens) for this round - never raises. Tokens are counted even
    on a round that fails to parse, since the call still cost real tokens and
    src/evaluation.py's cost reporting needs the true total, not just the
    successful round's."""
    agent = _build_critic_agent(instruction, model)
    runner = InMemoryRunner(agent=agent, app_name=APP_NAME)
    session = await runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
    content = genai_types.Content(role="user", parts=[genai_types.Part(text="Review the run described in your instructions.")])

    final_text = ""
    prompt_tokens = 0
    completion_tokens = 0
    async for event in runner.run_async(user_id=USER_ID, session_id=session.id, new_message=content):
        if event.usage_metadata is not None:
            prompt_tokens += event.usage_metadata.prompt_token_count or 0
            completion_tokens += event.usage_metadata.candidates_token_count or 0
        if event.is_final_response() and event.content and event.content.parts:
            final_text = "".join(part.text or "" for part in event.content.parts)

    return _parse_verdict(final_text), prompt_tokens, completion_tokens


def _fallback_verdict(static_findings: list[StaticFinding], evidence_categories: set[str]) -> _CriticVerdict:
    """Used when every round either fails to parse or is gated as an
    unevidenced reject - the critic must always return something, never
    silently skip review (ADR-008's principle: a code-enforced ceiling, not
    a request, with a defined result when that ceiling is hit).

    Gated on evidence_categories, not static_findings: static_findings also
    holds confirmation-only findings (e.g. "target column IS excluded") that
    are not evidence of a defect, so checking it directly would let an
    unevidenced reject survive relabelled as a fallback - exactly the
    hallucination the gate in critique_run_async exists to stop (ADR-016).
    The chosen category is real static evidence, never a hardcoded guess.

    The reported evidence text is filtered to findings tagged with the
    chosen category - on code that trips two categories at once (e.g.
    missing seed and missing stratify), the fallback must not name one
    category while its evidence text describes both."""
    if evidence_categories:
        category = sorted(evidence_categories)[0]
        matching_evidence = "; ".join(f.text for f in static_findings if f.category == category)
        return _CriticVerdict(
            verdict="reject",
            defect_category=category,
            defect=f"LLM critic review did not produce a usable, evidence-backed verdict; falling back to the deterministic static check, which found evidence of {category}.",
            evidence=matching_evidence,
        )
    return _CriticVerdict(
        verdict="accept",
        defect_category="none",
        defect="",
        evidence="LLM critic review did not complete (or only rejected without static evidence) and the deterministic static checks found no real defect evidence.",
    )


async def critique_run_async(
    report: RunReport,
    profile_text: str,
    static_findings: list[StaticFinding],
    model: str | LiteLlm,
) -> Critique:
    """static_findings is the single-pass result of src.checks.run_checks() -
    both the display text and the evidence-gate categories are derived from
    it here, rather than being computed twice by the caller. A reject
    naming a category with no matching finding - other than the
    UNGATED_CATEGORIES exemptions - is unevidenced: treated like an
    unparseable response and retried, closing the general hallucination
    class rather than trusting any reject the LLM states (ADR-016)."""
    evidence_categories = {f.category for f in static_findings if f.category is not None}
    finding_texts = [f.text for f in static_findings]
    instruction = _build_critic_instruction(report, profile_text, static_findings)
    verdict: _CriticVerdict | None = None
    rounds_used = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    gated_rejects = 0
    for rounds_used in range(1, MAX_CRITIQUE_ROUNDS + 1):
        verdict, prompt_tokens, completion_tokens = await _run_critic_round(instruction, model)
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        if (
            verdict is not None
            and verdict.verdict == "reject"
            and verdict.defect_category not in UNGATED_CATEGORIES
            and verdict.defect_category not in evidence_categories
        ):
            gated_rejects += 1
            verdict = None
        if verdict is not None:
            break
    fallback_used = verdict is None
    if verdict is None:
        verdict = _fallback_verdict(static_findings, evidence_categories)
    return Critique(
        verdict=verdict.verdict,
        defect_category=verdict.defect_category,
        defect=verdict.defect,
        evidence=verdict.evidence,
        static_findings=finding_texts,
        rounds_used=rounds_used,
        fallback_used=fallback_used,
        gated_rejects=gated_rejects,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
    )
