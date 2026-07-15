# PLAN OF APPROACH

-----

**Project**
Doggy Door: Perception for Robot-Door Interaction

**Date**
15-07-2026


**Context**
*Independent personal project (not a formal internship)*

---

## Introduction
This document covers the plan of approach for the project. due to circumstances im now doing this project by myself instead of with the original client. This document just scopes the project and lays order for what comes next. This will not be checked or aproved by the client.


---

## Problem Description

Many indoor robotic applications require autonomous mobile robots to open and close doors to navigate between rooms (search-and-rescue, security patrolling, care robots accessing patient rooms). Existing research on autonomous exploration assumes all doors are open, which rarely holds in practice. A robot that can't handle closed doors is severely limited in real-world deployments.

This project builds the **perception** layer (of the perception → cognition → action pipeline) needed for robot-door interaction: detecting doors and handles, classifying door properties, and estimating confidence: the foundation higher-level cognition/action modules can be built on.

---

## Goal
This project aims to develop a perception system that enables a robot to detect doors and door handles, classify their properties, and estimate the confidence of these outputs, laying the groundwork for a robot to later determine how to open them.

---

## Main Research Question

"What is the most effective approach to detecting, localizing, and classifying doors and door handles, with quantifiable confidence, in a simulated environment that reflects real-world conditions?"



### Sub questions

Draft scaffold based on the assignment's four challenge areas; refine wording and research methods once approach is settled:

1. **Simulation**: Which simulation environment can host a navigable indoor scene with a controllable mobile agent (robot or robot-equivalent camera rig) moving through it, such that the perception pipeline can be run on live sensor data from the agent's point of view as it approaches doors, and against what criteria (navigation/physics support, sensor simulation fidelity, ROS2 integration, scene/asset authoring effort, licensing/cost) should that environment be chosen?
   - Research method: Desk research comparing candidate simulators (e.g. NVIDIA Isaac Sim, Gazebo, Habitat, AI2-THOR, Unity + ML-Agents/ROS-TCP) against the criteria above, weighted toward ROS2 support since that's a listed required skill in the assignment brief.

2. **Door and window detection**: How can doors and windows be reliably detected and localized in both 2D image plane and 3D space, to support approach trajectory planning?
   - Research method: Desk research, survey existing door-detection approaches, architectures (e.g. YOLO-, Mask R-CNN-based detectors), and datasets (e.g. DeepDoors2) to select a starting architecture; then prototyping, implement and train/fine-tune the chosen detector on simulator-generated data and evaluate its 2D/3D localization performance.
3. **Door handle detection**: How can door handle location and type (lever vs. round knob) be detected, given that type determines manipulation strategy?
   - Approach: two-stage cascade. Reuse the door detector from Sub question 2 to localize and crop the door region, then run a dedicated second-stage model on the crop to localize the handle and classify its type. Chained by design: if the door isn't found, handle detection is skipped rather than attempted blind on the full scene.
   - Research method: Desk research, survey handle localization/classification approaches for the second-stage model; then prototyping, implement and train/fine-tune it on cropped door regions from simulator-generated data, evaluating localization accuracy and lever-vs-knob classification accuracy.

4. **Classification of door properties**: How can opening direction (inward/outward), glass doors, and sliding vs. standard doors be reliably classified?
   - Approach: piggybacks on the door crop from Sub question 2's detector (runs parallel to the Sub question 3 handle cascade, not chained after it). A classification model takes the same cropped door region and predicts opening direction, glass vs. solid, and sliding vs. standard.
   - Research method: Desk research, survey multi-label/multi-attribute classification approaches for the property set above; then prototyping, implement and train/fine-tune a classifier on cropped door regions from simulator-generated data, evaluating per-attribute classification accuracy.

5. **Confidence estimation**: How can detection/classification outputs be paired with a reliable confidence score, especially under low light, occlusion, or sensor noise?
   - Approach: no separate confidence model. Each of the three prior models (door/window detector, handle cascade, property classifier) outputs its own confidence score alongside its prediction (e.g. softmax/sigmoid probability or an equivalent per-architecture score), rather than a single pipeline-wide score.
   - Research method: Desk research, survey confidence/uncertainty estimation techniques suited to each model type (e.g. calibrated softmax, Monte Carlo dropout, ensembling); then prototyping, attach and calibrate confidence output on each of the three models and evaluate reliability under degraded conditions (low light, occlusion, sensor noise).


---

## Scope

**In scope:**
- Modular perception pipeline covering the four challenge areas above
- Experimental code repository (this repo)

**Out of scope:**
- using an actual robotics platform, due to lack of funding and space.(If offered defenitely doing it) currently focusing on simulation.

---

## Approach / Methodology

TBD

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
