# Doggy Door: Perception for Robot-Door Interaction

**Author:** Madelief Muntenaar
**Date:** draft, 2026-11-12
**Context:** Independent personal project. The technical scope comes from the Saxion-SMART assignment "Perception for Robot-Door Interaction" (v2, 2026-06-26); the original group is no longer involved.

> Draft v0. Sections 1 to 3 are a first draft. Sections 4 to 7 are an outline with notes on what goes in them.
> Citations are IEEE style, numbered in the order they first appear.

---

## 1. Introduction

This report covers Doggy Door, a project where I built the perception part of a robot that has to deal with closed doors. The robot needs to find doors, find the handle, work out what kind of door it is (push, pull or slide, which side the hinge is on, glass or not), and say how sure it is about each of those answers.

I did everything in simulation. A wheeled robot drives through generated buildings in Gazebo, looks for doors, and goes through them. There was no budget or space for a real robot, so the simulator also had to make the hard cases: smoke, dark rooms, and camera noise.

The report follows the research questions from my plan of approach. Chapter 2 explains the problem, chapter 3 lists the questions, chapter 4 explains how I worked, and chapter 5 answers each sub-question with its own method and results. Chapters 6 and 7 bring it together.

## 2. Problem Description and Analysis

Many indoor robots have to get from one room to another: search-and-rescue robots after a fire, security robots at night, care robots bringing medication to a patient. Most research on robots exploring buildings assumes that every door is open. In real buildings that is rarely true. Doors are closed for privacy, for security, to keep a fire contained, or just out of habit. A robot that can't handle a closed door can't reach a large part of the building.

A robot works in three steps: perception (sensing and understanding what is around it), cognition (deciding what to do) and action (actually doing it, like driving to the door and pushing it). This project is about the first step. The robot's actions can only be as good as what it perceives, so the perception has to give the next steps what they need:

- where the door is, in the camera image and in the room;
- where the handle is and what type it is, because a lever and a round knob need a different grip;
- how the door opens, since push, pull and slide each need a different approach;
- whether it is glass, since glass confuses depth cameras;
- and how sure the model is, so the robot can decide to act, take another look, or ask a person.

The last point matters most when conditions are bad. A model that says "95% sure" in thick smoke and is wrong half the time is worse than a model that admits it doesn't know.

## 3. Research Questions

**Main question:** What is the most effective approach to detecting, localizing, and classifying doors and door handles, with quantifiable confidence, in a simulated environment that reflects real-world conditions?

**Sub-questions:**

1. **Simulation.** Which simulation environment can host a building the robot can drive through, so the perception can run on live camera data as the robot approaches doors? Which criteria decide that choice?
2. **Door and window detection.** How can doors and windows be detected and located, both in the image and in 3D, well enough to plan a route to the door?
3. **Door handle detection.** How can the handle's location and type (lever or round knob) be detected?
4. **Door properties.** How can opening direction, glass doors and sliding doors be told apart reliably?
5. **Confidence.** How can each detection and classification come with a confidence score that can be trusted, also in low light, smoke and sensor noise?

## 4. Methodology

*Outline, to write:*

- Overall pipeline: camera frame → door/window detector → door crop → handle model and property model in parallel; every model gives its own confidence.
- Data: synthetic frames from generated rooms and buildings (door types, lighting, smoke, noise); which door types are held out of training to test on new door types.
- Models and tools: Gazebo Harmonic, ROS 2 Jazzy, Nav2, D-FINE detector, ResNet-18 property heads, PyTorch, MLflow for experiment tracking.
- Experiment design: train / val / test split by scene, held-out door types, degraded test conditions.
- Metrics: mAP and IoU for boxes, 3D position error, per-head accuracy, expected calibration error (ECE), and "sure means sure" (how often a ≥ 0.9 answer is right).

## 5. Results per sub-question

Each part uses the same layout: What is this · Why this method · How · Results · Conclusion · Discussion.

### 5.1 Simulation
*To write.* Simulator comparison, why Gazebo, the building generator, the robot platforms (lidar first, radar and a camera ring later because lidar fails in smoke), the door-step and patrol runs.

### 5.2 Door and window detection
*To write.* Detector choice, training data versions, 2D results, 3D position from depth, results in the patrol runs (doors found per building).

### 5.3 Door handle detection
*To write.* Handle model, lever vs. knob accuracy, hinge side from handle position.

### 5.4 Door properties
*To write.* Property heads (opening, hinge side, glass, state), accuracy on known and new door types; the reactive door step that checks the guess by trying the door.

### 5.5 Confidence
*Draft notes (numbers from 2026-11-12, door property model `props_door_v7_dha_448`, test set 2305 doors):*

- Detectors: calibrated per class with temperature scaling [1] plus box-aware calibration [2]; detection ECE about 0.01.
- Property heads: one temperature per head, fitted on the validation set [1].
- Glass: well calibrated everywhere (ECE 0.02; still 0.07 on new door types and thick smoke).
- Opening and hinge side, known door types: good (ECE 0.05 to 0.07; a ≥ 0.9 answer is right 95 to 96% of the time).
- Opening and hinge side, new door types: the model is about 81% sure on average but right only 43 to 45% of the time (ECE 0.37). A ≥ 0.9 answer is right only 65% of the time.
- Thick smoke (visibility under 4 m): accuracy drops to 45%, confidence to 66 to 75% (ECE 0.21 to 0.29).
- Darkness affects calibration less than smoke does (ECE 0.13 dark vs. 0.09 normal light for opening).
- Fitting the temperature per smoke level fixes the smoke part (ECE down to 0.03), but no rescaling fixes new door types (ECE stays around 0.23). The model does not know when a door type is unfamiliar. This matches what is reported in the literature on calibration under dataset shift [3].
- Next: *open choice* (see the project note).

## 6. Discussion

*To write.* Simulation-to-real gap; what didn't work (hinge-end test, panel-angle filter, first SLAM patrol); false doors; lidar in smoke → radar; limits of the held-out test.

## 7. Conclusion

*To write.* Answer to the main question.

## References

[1] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in *Proc. 34th Int. Conf. Machine Learning (ICML)*, 2017, pp. 1321–1330.

[2] F. Küppers, J. Kronenberger, A. Shantia, and A. Haselhoff, "Multivariate confidence calibration for object detection," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition Workshops (CVPRW)*, 2020, pp. 326–327.

[3] Y. Ovadia *et al.*, "Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 32, 2019.

## Appendices

*To add:* config dumps, extra tables, building maps.
