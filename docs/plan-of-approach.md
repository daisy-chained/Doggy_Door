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
This document covers the plan of approach for the project. Due to circumstances I'm now doing this project by myself instead of with the original client. This document just scopes the project and lays order for what comes next. This will not be checked or approved by the client.


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
- using an actual robotics platform, due to lack of funding and space. (If offered, definitely doing it.) Currently focusing on simulation.

---

## Approach / Methodology

The pipeline runs on RGB(-D) frames streamed live from a simulated agent (simulator TBD per Sub question 1) as it navigates an indoor scene, optionally paired with LiDAR point cloud data as an additional sensing modality (particularly useful for 3D localization). Each frame passes through the door/window detector (Sub question 2); on a positive detection, the cropped door region feeds two parallel branches: the handle cascade (Sub question 3) and the property classifier (Sub question 4). All three models attach their own confidence score to their output (Sub question 5).

**Data:** simulator-generated synthetic frames as the primary training/eval set, supplemented by public door-detection datasets identified during Sub question 2's desk research (e.g. DeepDoors2) if useful for pretraining or comparison.

**Frameworks/tools:** Python, PyTorch, ROS2 (for agent control and sensor streaming, if the chosen simulator supports it).

**Evaluation metrics:** mAP/IoU for door-window 2D localization and a 3D localization error metric; localization and classification accuracy for handles (lever vs. knob); per-attribute accuracy for door properties; calibration quality (e.g. expected calibration error) for confidence scores, tested specifically under degraded conditions (low light, occlusion, sensor noise).

---

## Planning

| Phase | Description | Timeframe |
|---|---|---|
| 1. Research | Desk research across all 5 sub-questions (simulator options, detection/handle/classification approaches, confidence techniques) plus light feasibility spikes; produces the report's Approach/Methodology | 2026-07-15 to 2026-09-02 (7 wks) |
| 2a. Build: detection | Implement, train, evaluate door/window detector | 2026-09-02 to 2026-09-30 (4 wks) |
| 2b. Build: handle cascade | Implement, train, evaluate handle localization and classification | 2026-09-30 to 2026-10-21 (3 wks) |
| 2c. Build: property classification | Implement, train, evaluate door property classifier | 2026-10-21 to 2026-11-11 (3 wks) |
| 2d. Build: confidence estimation | Attach and calibrate confidence output across all three models | 2026-11-11 to 2026-11-25 (2 wks) |
| 2e. Integration and evaluation | Integrate into one pipeline; run the agent through the full simulated building under varied environmental conditions | 2026-11-25 to 2026-12-30 (5 wks) |
| 2f. Reporting and demo | Write final report, record video demonstration | 2026-12-30 to 2027-01-31 (~4.5 wks) |

Deadline: 2027-01-31, a simulated agent navigating a full building under varied environmental factors, demonstrating the full pipeline reliably.

---

## Stakeholder Analysis

Single-stakeholder project. There's no external party to analyze.

| Stakeholder | Interest | Power | Notes |
|---|---|---|---|
| Project owner | High | High | Executes the project |

---

## Risk Analysis

| Risk | Category | Probability | Impact | Mitigation |
|---|---|---|---|---|
| Sim-to-real gap: models trained only on simulated data may not generalize to real photos (lighting, textures, sensor noise differ) | Research | Medium-High | High: undermines validity of results if never checked against reality | Use domain randomization in the simulator (varied lighting, textures, materials); if any real images are obtainable, spot-check against them and document the gap as an explicit limitation if not closed |
| Chosen simulator turns out to lack a needed capability (e.g. ROS2 integration, sensor fidelity) discovered after committing engineering time | Research | Low-Medium | Medium: could force a late simulator switch | Front-load Sub question 1's desk research fully before starting any model work |
| Limited variety in simulator-generated data leads to overfitting | Research | Medium | Medium | Procedurally vary scenes, door types, materials, and lighting during data generation |
| Solo project with no external deadline or accountability, risk of stalling or drifting scope | Project | Medium-High | Medium: project quietly stops making progress | Self-imposed milestones/timeframes (tie to Planning section); keep dev-log entries per session |
| Compute/hardware constraints training multiple models (detector, handle cascade, classifier) plus running a simulator | Project | Medium | Medium: slows iteration | Use lightweight/pretrained backbones; scale down resolution/dataset size if needed |
| Scope creep across four semi-independent modules plus the open-ended simulator choice | Project | Medium | Medium: never reaches "done" | Define a "good enough" threshold per evaluation metric upfront (from Approach/Methodology), timebox each phase |

---

## Deliverables

The assignment brief dictates the first 3 deliverables, the others are added to mitigate some of the risks mentioned above.

1. A report detailing the approach and results.
2. A code repository with the experimental code (this repo).
3. A video demonstration of the results.
4. A Definition of done.
5. A dev log summary.
6. An evaluation on the project.

---

## Related

- Assignment brief: `docs/assignment-brief.md`
- Vault project note: [[Projects/Doggy Door]] (personalBrain)
