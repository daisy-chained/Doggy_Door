# PLAN OF APPROACH

-----

**Project**
Doggy Door — Perception for Robot-Door Interaction

**Date**
TBD

**Context**
*Independent personal project (not a formal internship)*

---

## Introduction

TBD — frame what this document covers, what the project is, and why it matters, before diving into the problem. See `docs/assignment-brief.md` for the source assignment context.

---

## Problem Description

Many indoor robotic applications require autonomous mobile robots to open and close doors to navigate between rooms — search-and-rescue, security patrolling, care robots accessing patient rooms. Existing research on autonomous exploration assumes all doors are open, which rarely holds in practice. A robot that can't handle closed doors is severely limited in real-world deployments.

This project builds the **perception** layer (of the perception → cognition → action pipeline) needed for robot-door interaction: detecting doors and handles, classifying door properties, and estimating confidence — the foundation higher-level cognition/action modules can be built on.

---

## Goal

TBD — one clear goal statement for what this project delivers by the end.

---

## Main Research Question

TBD

### Sub questions

Draft scaffold based on the assignment's four challenge areas — refine wording and research methods once approach is settled:

1. **Door and window detection** — How can doors and windows be reliably detected and localized in both 2D image plane and 3D space, to support approach trajectory planning?
   - Research method: TBD (desk research / experimentation with detection models)

2. **Door handle detection** — How can door handle location and type (lever vs. round knob) be detected, given that type determines manipulation strategy?
   - Research method: TBD

3. **Classification of door properties** — How can opening direction (inward/outward), glass doors, and sliding vs. standard doors be reliably classified?
   - Research method: TBD

4. **Confidence estimation** — How can detection/classification outputs be paired with a reliable confidence score, especially under low light, occlusion, or sensor noise?
   - Research method: TBD

---

## Scope

**In scope:**
- Modular perception pipeline covering the four challenge areas above
- Experimental code repository (this repo)

**Out of scope:**
TBD

---

## Approach / Methodology

TBD — datasets, models/frameworks (Python, PyTorch and/or TensorFlow, ROS2, 3D computer vision techniques), experiment design, evaluation metrics.

---

## Planning

| Phase | Description | Timeframe |
|---|---|---|
| 1. Research & setup | Literature review, environment/tooling setup, dataset scoping | TBD |
| 2. Door & window detection | Build and validate detection component | TBD |
| 3. Door handle detection | Build and validate handle detection component | TBD |
| 4. Door property classification | Build and validate classification component | TBD |
| 5. Confidence estimation | Build and validate confidence scoring across components | TBD |
| 6. Integration & evaluation | Integrate modules into a single pipeline, evaluate end-to-end | TBD |
| 7. Reporting & demo | Write final report, record video demonstration | TBD |

Actual timeline TBD — this runs as an independent personal project, not a formal internship placement.

---

## Stakeholder Analysis

TBD

| Stakeholder | Interest | Power | Notes |
|---|---|---|---|
| Project owner | High | High | Executes the project |

---

## Risk Analysis

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| TBD | TBD | TBD | TBD |

---

## Deliverables

Per the assignment brief:

1. A report detailing the approach and results.
2. A code repository with the experimental code (this repo).
3. A video demonstration of the results.

---

## Related

- Assignment brief: `docs/assignment-brief.md`
- Vault project note: [[Projects/Doggy Door]] (personalBrain)
