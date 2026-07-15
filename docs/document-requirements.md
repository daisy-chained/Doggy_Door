# Document Requirements: Doggy Door Deliverables

**Project:** Doggy Door: Perception for Robot-Door Interaction
**Date:** 2026-07-15
**Sources:**
- `docs/assignment-brief.md`, defines the first 3 deliverables
- `docs/plan-of-approach.md`, adds 3 more deliverables (Definition of Done, dev log summary, project evaluation)
- Saxion HBO-ICT Report Scan standard (year-3 level), defines the *quality bar* the written deliverable is held to here

This document exists separately from `docs/plan-of-approach.md` because the Plan of Approach describes the *project*; this describes the *deliverables*: what each required output must contain and what standard they must meet before being considered "done."

---

## Why a Saxion-standard quality bar

This project is explicitly **not** a formal internship (see `docs/assignment-brief.md` and the project's vault note). The originating group is no longer involved, and it runs as an independent personal project with Saxion-SMART's assignment kept only for technical scope. There is no internship coordinator, no second assessor, and no Brightspace submission.

The report-scan standard is applied anyway, by choice, as the quality bar for the written deliverable, the same bar Saxion HBO-ICT applies to year-3 professional documents. No submission deadline or assessor applies; only the content and formatting rules below do.

---

## 1. Report: "detailing the approach and results"

### 1.1 Required structure

The report should follow a research-report structure, since the project is organized around a main research question and sub-questions (already scaffolded in `docs/plan-of-approach.md`):

| Section | Content |
|---|---|
| Cover page | Title, author, date, project context (personal project, Saxion-SMART assignment as source) |
| 1. Introduction | What the report covers, what the project is, why door perception matters (robot-door interaction, closed-door assumption gap) |
| 2. Problem Description and Analysis | The gap being addressed: perception pipeline for closed doors, position in the perception → cognition → action pipeline |
| 3. Research Questions | Main research question + the sub-questions (simulation, door/window detection, door handle detection, door property classification, confidence estimation) |
| 4. Methodology | Overall approach: datasets, models/frameworks, experiment design, evaluation metrics |
| 5. Results, per sub-question | For **each** sub-question: What is this / Why this method / How this method / Results / Conclusion / Discussion |
| 6. Discussion | Cross-cutting findings, limitations, what didn't work and why |
| 7. Conclusion | Answer to the main research question, summary of contributions |
| References | All cited work (papers, datasets, frameworks) |
| Appendices | Supplementary detail: extra figures, tables, config dumps |

### 1.2 Content requirements per deliverable challenge

Each of the four challenge areas from `docs/assignment-brief.md` must be traceable to a specific section of Results (§1.5 above):

1. Door and window detection: 2D + 3D localization results
2. Door handle detection: location + type (lever vs. round knob) results
3. Classification of door properties: opening direction, glass, sliding vs. standard
4. Confidence estimation: per-detection/classification confidence, tested under at least one degraded condition (low light, occlusion, or sensor noise)

If the "simulation" sub-question (currently unfinished in the Plan of Approach) results in a simulated environment being used for data generation or evaluation, that must also get its own subsection describing what was simulated and why.

### 1.3 Formatting / quality requirements (report scan, applied)

| Requirement | Standard |
|---|---|
| Filename & format | Consistent, versioned filename (e.g. `Doggy_Door_Report_v1.pdf`); PDF export of the final version |
| Cover page | Author name (Madelief Muntenaar) and date present |
| Completeness | All sections in §1.1 present; no placeholder/TBD sections in the final version |
| Citation style | One consistent standard throughout: APA, IEEE, or Harvard. **Recommendation: IEEE**, since it's the standard for CV/ML papers this report will cite (YOLO, Mask R-CNN, etc.) |
| Formatting | Consistent font and heading/chapter layout throughout |
| Language quality | ≤5 spelling/grammar errors per 450 words (year-3 standard) |
| Originality | No plagiarism: all external methods, datasets, and models properly cited, not just linked |

---

## 2. Code repository: "with the experimental code"

### 2.1 Required structure

The assignment brief calls for **modularity** explicitly: "it allows individual components to be developed, tested, and improved independently." The repo structure should reflect that directly: one module per challenge area, not a single monolithic script/notebook.

| Requirement | Detail |
|---|---|
| Modular layout | Separate, independently runnable components for: door/window detection, handle detection, property classification, confidence estimation |
| README | Setup instructions, how to run each module, how to reproduce reported results |
| Reproducibility | Pinned dependencies (`requirements.txt` / environment file), documented dataset sources |
| Experiment record | Where results/metrics from the report are generated from: scripts or notebooks that produce the report's numbers, not just the numbers pasted in |
| Confidence output | Every detection/classification module must expose a confidence score in its output, not just a final report-time computation |
| License | Existing "All Rights Reserved" notice in `README.md` stays as-is |

### 2.2 Traceability to the report

Every result quoted in the report (§1) must be reproducible from a specific script/notebook path in this repo. The report should reference those paths directly rather than describing results in the abstract.

---

## 3. Video demonstration: "of the results"

| Requirement | Detail |
|---|---|
| Length | ~5-10 minutes: long enough to show each pipeline component, short enough to stay watchable |
| Content | Pipeline overview, then a demonstration of each of the four challenge areas in action (detection, handle ID, classification, confidence scores visibly displayed) |
| Narration | Spoken or captioned explanation of what's being shown at each step, not silent footage |
| Failure cases | At least one example of a low-confidence or failed detection, showing the confidence-estimation component doing its job |
| Format | Standard video file or unlisted link, referenced from the report and/or README |

---

## 4. Definition of Done

Deliverable completeness, not metric thresholds. The project is done when all 6 deliverables from `docs/plan-of-approach.md` exist and meet the requirements defined in this document:

| Deliverable | Done when |
|---|---|
| Report | Meets all of §1 (structure, content, formatting) |
| Code repository | Meets all of §2 (modular layout, README, reproducibility, traceability) |
| Video demonstration | Meets all of §3 |
| Definition of Done | This section exists and every row in this table is checked off |
| Dev log summary | Meets §5 |
| Project evaluation | Meets §6 |

---

## 5. Dev Log Summary

A running log covering both project phases (Planning §1, §2a-2f), not just build sessions.

| Requirement | Detail |
|---|---|
| Research-phase entries | One entry per significant desk-research session or decision during Phase 1 (e.g. why a given simulator/architecture was chosen over alternatives, findings from surveyed papers/datasets) |
| Build-phase entries | One entry per significant build/training session during Phase 2 (2a-2f), what was attempted, what worked/didn't, and why |
| Rationale, not just activity | Each entry records the reasoning behind a decision, not just what was done, so the report's Methodology and Discussion sections can draw on it directly |
| Format | Dated entries, chronological, kept in this repo (e.g. `docs/dev-log.md` or per-session files) |

---

## 6. Project Evaluation

Process evaluation and reflection, not just a results recap.

| Requirement | Detail |
|---|---|
| Process evaluation | How the project ran against the Planning timeline (§Planning), which Risk Analysis items actually materialized and how they were handled, what would be scoped differently next time |
| Reflection | What was learned technically (methods, tools, sim-to-real gap findings) and about ways of working solo without a client/deadline pressure |
| Outcome vs. research question | An honest assessment of how well the final pipeline actually answers the Main Research Question, not just a metrics recap |
| Format | Written section, part of or alongside the final report; no fixed method required (STARR/Korthagen not needed here, this isn't a formal internship reflection) |

---

## Related

- `docs/assignment-brief.md`, source of the three original deliverables and four challenge areas
- `docs/plan-of-approach.md`, project-level plan; sub-questions map 1:1 to Report §1.2, and Deliverables §4-6 correspond to this document's §4-6
