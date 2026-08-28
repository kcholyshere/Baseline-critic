# The Baseline Model Agent
Draft, expected to change.

## Business questions
The problems this exists to answer, in the order a client would ask them.
- Can this dataset predict the thing we care about at all, and how fast can we know?
- Is that score real, or is the model leaking and about to embarrass us in production?
- How many analyst days go into first baselines that get thrown away regardless?
- Can someone who is not a modeller get a first baseline they are allowed to trust?
- What did we already try on this dataset, and why was it rejected?

Cost optimisation, not revenue. The saving is analyst days per dataset, plus the
avoided cost of shipping a model whose score was never real - which is the
expensive one, because it is found late and by someone else.

## What
Point it at a tabular dataset and a target column. It profiles the data, proposes
and builds features, trains a handful of baselines under a fixed budget, and hands
back the winning model with a written account of what it tried.

The part that makes it a project rather than AutoML: a **separate critique agent
reviews every result before it is accepted**, and its job is specifically to be
suspicious of good news. Leakage, train/test contamination, a target column
smuggled into the features, temporal leakage, unseeded randomness. A score is not
a result until the critic has failed to explain it away.

## Why
Anyone can wire an LLM to scikit-learn. The interesting design problem is the
reviewer: an agent whose success is measured by rejecting work, given different
information from the agent that produced it, and bounded so it cannot argue
forever. That is a sharper version of the critique loop and it has a real job to
do rather than a cosmetic one.

## Functionalities
- Profile a dataset: types, cardinality, missingness, target balance, obvious keys.
- Propose features in plain language, then implement them as code.
- Train and score several baselines under a compute budget.
- Critique each accepted result for leakage and contamination, and reject with a reason.
- Report what was tried, what won, and what was rejected and why.
- Emit a runnable script or notebook that reproduces the winning result from scratch.

## Baseline architecture
An orchestrated loop, one dataset per run:
`profile -> propose -> train -> score -> critique -> accept or revise`

| Component | Role |
|---|---|
| Planner/modeller agent | decides what to try next, writes the feature and training code |
| Critic agent | reviews a scored result for leakage and contamination; accepts or names one defect |
| Code execution tool | runs generated Python in a working directory, returns stdout and artefacts |
| Profiling tool | deterministic dataset summary, so the model never guesses at schema |
| Scoring tool | trains and evaluates against a fixed split the agents never see |
| Report tool | renders the run into a document (the Canvas pattern, reused) |

Two budgets, same two-tier idea as before: a hard ceiling in code on training runs
per session, and a soft per-request budget on critique rounds.

## Tech stack
- Python, ADK for the agents, Gemini via Vertex AI.
- pandas or polars for data, scikit-learn plus LightGBM for baselines.
- Code execution in a subprocess with a restricted working directory. Keep it dumb
  and local first; containerise later if it matters.
- Langfuse for tracing, since diagnosing an agent from traces already proved itself.
- Run records as plain JSON, same as the current evaluation harness.
- Optional later: the critic as an A2A service, dataset access over an MCP server.

## How it is evaluated
This is the strongest part of the idea and worth leading with.
- **The oracle is free.** A held-out split the agent never sees gives a real number,
  so there is no LLM judge anywhere.
- **The critic is tested by injecting leaks.** Plant a known leak in a clean dataset
  and measure whether the critic catches it, against how often it cries wolf on a
  clean run. Detection rate and false alarm rate, both countable.
- **Cost is tracked per dataset**: tokens, wall clock, and number of training runs.
- Everything is deterministic and re-scorable from stored runs.

## Challenges
- **Sandboxing.** The agent writes and runs code. Deciding what it may do unsupervised
  is the main safety design, and it is a real decision rather than a checkbox.
- **Gaming the metric.** Nothing stops an agent fitting the holdout if it can see it.
  The split has to be enforced structurally, not asked for politely.
- **Telling "too good" from "genuinely easy".** The critic will produce false alarms
  on datasets that are simply separable. Calibrating that is the hard part.
- **Context.** A dataset does not fit in a prompt. Everything the model sees about
  the data is a summary, and choosing what goes in the summary is a design decision.
- **Latency.** Training takes real time, so a turn is minutes rather than seconds.

## Advantages
- Free, uncontestable ground truth, which is what made the last evaluation credible.
- Genuinely useful: the first honest baseline is real work, and leaked models do
  reach production.
- Reuses the critique loop and the artefact tool, so the new engineering is the
  reviewer and the sandbox rather than scaffolding.
- Answers the business questions above directly, and the expensive one - a leaked
  score reaching production - is the whole job of the critic.

## Out of scope for a first version
Deep learning, large hyperparameter searches, deployment, multi-dataset comparison,
and time series. Tabular classification and regression only.

## Suggested phases
1. One dataset, one baseline, a report. End to end and boring.
2. The critic, with leakage detection as its only job.
3. The feature proposal loop, so the agent revises rather than runs once.
4. The evaluation harness, with injected leaks and a cost budget.
5. Optional: the critic as its own A2A service, dataset access over MCP.
