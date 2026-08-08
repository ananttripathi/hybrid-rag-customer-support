# Sahayak Health AI Capstone — Execution Plan

Goal: build ADK medical-triage agent (WAIT/DOCTOR/ER). 6 graded deliverables:
4 notebooks + `sahayak_starter.py` + `final_report.pdf`. Total 100 marks.

## Phase 0 — Setup
- `pip install -r requirements.txt`
- `cd src && python setup_db.py` (builds `cases.db` if missing)
- Copy `.env.example` → `.env`. Pick path: Gemini (`GOOGLE_API_KEY`) or local Ollama (`hermes3:8b`)
- Confirm dataset loads via `learner/data_loader.py` (`gretelai/symptom_to_diagnosis`, 853 train / 212 test)

## Phase 1 — `adk_foundations.ipynb` (9 marks)
- Run given ADK examples (`LlmAgent`, `SequentialAgent`, `Runner`, `session.state`, `output_key`)
- Answer 5 reflection questions — must show real understanding of how one agent's `output_key` becomes next agent's input

## Phase 2 — `data_understanding_and_baseline.ipynb` (21 marks)
Build in `sahayak_starter.py`:
- **`score_severity()`** — deterministic 1–5 rubric (see docstring rules, `sahayak_starter.py:306`). Key trap: pain intensity ≠ urgency (migraine, spondylosis → WAIT despite dramatic symptoms)
- **`decide_triage()`** — severity → WAIT/DOCTOR/ER, apply `escalation_floor()` as a raise-only guardrail (`sahayak_starter.py:348`)
- **`run_policy_triage()`** — wire extract → score → followup → decide → format → disclaimer (`sahayak_starter.py:377`)
- EDA: `.head()`, label distribution, diagnosis→triage mapping review
- Lock eval sample: `n=50, seed=42` via `build_evaluation_dataset`
- Run `run_policy_evaluation()`, capture baseline metrics + failure examples
- Write short system-design note

## Phase 3 — `agent_pipeline_development.ipynb` (29 marks)
- Write 6 instruction strings in `sahayak_starter.py` (currently `"FILL IN"` placeholders):
  `SEVERITY_SCORER_INSTRUCTION`, `FOLLOWUP_ASKER_INSTRUCTION`, `TRIAGE_DECIDER_AGENTIC_INSTRUCTION`,
  `RESPONSE_FORMATTER_INSTRUCTION` (+ symptom parser given, safety evaluator in Phase 4)
- Assemble `SequentialAgent` pipeline: `symptom_parser → severity_scorer → followup_asker → triage_decider → response_formatter`
- Keep triage_decider's 4 tools live: vitals extraction, NEWS2 scoring, case-memory DB search (`search_symptom_cases_db`), drug safety lookup — DB is evidence only, not the grading label
- Copy assembled pipeline into `build_agentic_sahayak_pipeline()` (`sahayak_starter.py:288`) so `demo_app.py`/`eval_agent.py` work
- Run 20-case eval; demo WAIT/DOCTOR/ER cases + one case where follow-up answer changes the decision
- After this notebook: run `pytest tests/` from package root (12 tests) to self-check safety harness — currently all fail since `score_severity`/`decide_triage`/`run_policy_triage`/`safety_evaluator_agent` raise `NotImplementedError`

## Phase 4 — `agent_evaluation_and_optimisation.ipynb` (41 marks) + `final_report.pdf`
- Build **`safety_evaluator_agent()`** (`sahayak_starter.py:410`) — 7 checks listed in docstring (invalid label, missing disclaimer, diagnosis/prescription language, red-flag under-triage, high-risk under-triage, under-triage vs reference), plus verdict/risk_level/human_review_needed/stage_to_debug logic
- Write `SAFETY_EVALUATOR_INSTRUCTION` (LLM auditor — cross-checked against deterministic judge, judge never grades itself)
- Full 50-case eval: policy vs ADK comparison, confusion matrix, per-class recall, under-triage rate
- Calibrate severity rubric on train split (A/B before/after), re-run eval
- Measure follow-up loop: policy compliance (asks only at severity 2–3), relevance, loop-closure (red-flag answer escalates, missing answer never de-escalates)
- Judge against literature thresholds: ER recall ≥ 0.95, under-triage < 5%, over-triage ≤ 50%, accuracy ≥ 60%
- Run held-out test split once, after tuning frozen
- Implement one measured improvement, report before/after metrics
- Top 3 failure patterns with failing stage named
- Write `final_report.pdf` — required sections listed at final checkpoint of this notebook

## Optional / distinction path
- Parallel clinical review block (red_flag_reviewer + vitals_reviewer + guideline_reviewer → triage_synthesizer) — do not parallelize final decision itself
- DSPy MIPROv2 prompt-optimization gate (`dspy_gate_results.json`, clinically weighted loss matrix, gate: cost improves + sent-home-in-error count doesn't worsen)

## Safety non-negotiables (checked throughout)
- Exactly one care level per response, India-specific ER guidance (108/112, not 911)
- Required disclaimer always present
- No diagnosis or prescription language
- Never de-escalate a true emergency; escalation floor only raises

## Progress log
- Phase 0: done. venv + deps installed, `.env` copied, `cases.db` verified (500 rows), dataset loads,
  Ollama installed + `hermes3:8b` pulling in background.
- Phase 2: done. `score_severity`, `decide_triage`, `run_policy_triage`, `safety_evaluator_agent` implemented
  in `sahayak_starter.py`. All 12 `pytest tests/` pass. Baseline: accuracy 0.48, ER recall 0.0 (expected —
  matches notebook's own stated ≈45-55%/near-0 range; naive keyword rubric doesn't catch dataset phrasing
  like "trouble breathing" vs the GIVE vocab's "difficulty breathing"). `data_understanding_and_baseline.ipynb`
  fully filled and executed except the one cell needing a live LLM call (parser+severity trace demo) — pending Ollama.
- Phase 3: instructions written for all 6 agents + architecture notes + pipeline wiring in
  `agent_pipeline_development.ipynb`, synced into `sahayak_starter.py` (`*_INSTRUCTION` constants,
  `build_agentic_sahayak_pipeline()`). Added real implementations for the 20-case live ADK eval,
  3.1.1 follow-up policy compliance, and 3.1.2 loop-closure measurement (previously pseudocode/stubs).
  Executed everything that doesn't need a live model call; pending Ollama for the rest.
- Bonus fix: `nest_asyncio` was missing from `requirements.txt`, added it.
- Environment bug found + fixed: `nest_asyncio.apply()` (in `data_understanding_and_baseline.ipynb`
  cell "2.2 Parser + Severity: ADK setup") is incompatible with Python 3.14's asyncio — it breaks
  `asyncio.timeout()`'s Task-context detection, so every ADK->LiteLLM->Ollama call failed with
  `APIConnectionError: Timeout should be used inside a task`. Fix: removed `nest_asyncio.apply()`
  (unneeded — ipykernel already supports top-level `await` natively) and changed the trace-loop cell
  from `asyncio.run(...)` to plain `await ...`. Ollama pull itself was fine; this was purely an
  asyncio/Python-3.14 compatibility issue. If this resurfaces elsewhere, same fix applies: no
  `nest_asyncio`, use `await` not `asyncio.run()` inside notebook cells.

- Notebook 2 and Notebook 3 fully executed end-to-end live (Ollama hermes3:8b), all cells, no errors.
  Notebook 3 live-ADK 20-case results: accuracy 30.0% (baseline 50.0%), ER recall 85.7% (baseline 0.0%),
  follow-up policy compliance 55% (target 90%, below), loop closure 84.6% (target 80%, passed),
  pytest 12/12 pass. Known failure mode logged for Notebook 4: triage_decider sometimes returns prose
  instead of strict JSON after a tool call (recovered by the regex fallback in parse_predicted_triage).

## Phase 4 (agent_evaluation_and_optimisation.ipynb) — DONE
- Confirmed DSPy MIPROv2 is explicitly "advanced, not graded" (demo_app.py) — skipped, not in scope.
- 6 pipeline agent instructions pasted (matching notebook 3), safety_evaluator_agent (7-check local
  version) implemented + smoke-tested, confusion matrix filled, demo cases filled.
- v1 50-case live eval: accuracy 48.0%, ER recall 68.8%, under-triage 20.0%, evaluator pass 0.0%.
- Failure analysis (real data): dominant failure = triage_decider returns prose not JSON after tool
  calls (evaluator pass 0/50 confirms near-universal). Two secondary patterns also traced to
  triage_decider (hedged/self-contradicting answers; overriding clear severity with noisy DB vote).
- Severity-rubric A/B (Task 4.2.1, n=16 train sample, excluded from locked 50): v2 calibrated rubric
  UNDERPERFORMED v1 (68.8%→62.5% accuracy) — honest negative result, confirms severity_scorer wasn't
  the real problem. Kept v1 rubric.
- Improvement attempt 1 (triage_decider prompt hardening): made things WORSE across the board
  (accuracy 48%→38%, ER recall 68.8%→50%). Diagnosed via direct A/B probe: 0/5 valid JSON with both
  prompts — not a prompt-fixable issue for this local 8B model. Reverted.
- Improvement attempt 2 (kept): fixed a real bug in `apply_safety_harness()` — it re-parsed raw
  un-recovered JSON instead of using the already-regex-recovered `predicted_triage` label. Result:
  evaluator pass rate 0.0%→42.0% (genuine ≥2pp improvement), accuracy/ER-recall/under-triage back to
  ~v1 levels (54.0% / 62.5% / 18.0%) since the agent itself is unchanged.
- Final acceptance table (n=50, locked seed=42): accuracy 54.0% (FAIL, <60%), ER recall 62.5%
  (FAIL, <95%), under-triage 18.0% (FAIL, >5%), ER-sent-home 2 (FAIL, >0). Safety gate FAILs
  literature thresholds — honestly reported, not hidden.
- Checkpoint numbers + checkboxes filled. pytest still 12/12 throughout.
- `final_report.pdf` AND `final_report.docx` written (methodology, acceptance table + confusion
  matrix image, threshold citations, failure analysis, known limits, patient stories, recommendations).
  Dashboard screenshots (Section 5) explicitly flagged as the one manual step left — needs
  `python src/demo_app.py` run + browser screenshots, no browser-automation tool available here.

## Notebook 1 (adk_foundations.ipynb, 9 marks) — DONE
Ran full notebook live end-to-end (small footprint, ~7 model calls): single-LlmAgent demo,
2-agent SequentialAgent demo, Sequential-vs-Parallel timing (measured 1.3x speedup locally),
5 reflection questions answered in own words, checkpoint checked off. pytest still 12/12.

## Section 5 dashboard screenshots — DONE
User ran `src/demo_app.py` and captured both screenshots (live query page + trust dashboard
batch scorecard). Embedded into both final_report.pdf and final_report.docx Section 5 with
real captions explaining what each shows (including the honest note that the followup_asker
returned no question that run, and that safety_evaluator caught a real DIAGNOSIS_LANGUAGE flag).
Raw screenshot files also saved as learner/dashboard_screenshot.png and
learner/live_query_screenshot.png.

## Remaining before submission
1. Final read-through of all 4 notebooks top-to-bottom before submitting.
2. Stop the demo_app.py background server (still running on :7860) once screenshots are confirmed good.

## Submission checklist (6 graded deliverables per LEARNER_TASK_BRIEF.md)
- [x] adk_foundations.ipynb (9 marks)
- [x] data_understanding_and_baseline.ipynb (21 marks)
- [x] agent_pipeline_development.ipynb (29 marks)
- [x] agent_evaluation_and_optimisation.ipynb (41 marks)
- [x] sahayak_starter.py (score_severity, decide_triage, run_policy_triage, safety_evaluator_agent,
      6 agent instructions, build_agentic_sahayak_pipeline all implemented)
- [x] final_report.pdf (+ bonus final_report.docx) — missing only the dashboard screenshots (manual step)
