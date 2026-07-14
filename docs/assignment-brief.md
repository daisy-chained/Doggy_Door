# Assignment Brief — Perception for Robot-Door Interaction

**Version:** v2, June 26, 2026

*Transcribed from the original assignment PDF, kept in the vault (personalBrain) alongside the project note. Stored here so the Plan of Approach has its source context in-repo. Originating group is no longer involved in this project — kept for the technical scope only.*

---

## Introduction

Many indoor robotic applications require autonomous mobile robots to open and close doors in order to navigate between rooms — e.g. search-and-rescue robots entering rooms after a fire or structural collapse, security robots patrolling outside working hours, or care robots accessing a patient's room to deliver medication or monitor vital signs. The ability to detect, approach, and interact with doors is a fundamental requirement for effective, truly autonomous operation.

Prior research has looked at autonomous exploration with mobile robots in indoor emergency scenarios, where robots must navigate unknown environments quickly and reliably. Building on those insights, the aim is to apply the same principles to generate detailed building maps for planning patrolling or inspection routines across other domains (logistics facilities, hospitals, office buildings) — autonomously generated maps reduce manual surveying effort and let robots keep maps current as environments change.

Until now, a key simplifying assumption has always been made: that all doors are open. This rarely holds in practice — doors may be closed for privacy, security, fire containment, or simply routine. A robot that can't handle closed doors will be unable to access large portions of a building. Addressing this is a critical step toward truly autonomous mobile robots in real-world indoor environments.

Any autonomous robotic system relies on a pipeline of three interconnected components: **perception**, **cognition**, and **action**. Perception is the robot's ability to sense and interpret its environment via onboard sensors (cameras, depth sensors). Cognition is the reasoning/decision-making that turns sensory information into situational understanding. Action is the physical execution (navigating to a door, manipulating a handle). These are tightly coupled — the quality of the robot's actions is only as good as the quality of its perception.

---

## The Assignment

Develop a **modular perception pipeline** specifically for robot-door interactions. Modularity is deliberate — it allows individual components to be developed, tested, and improved independently, and the pipeline becomes the foundation for higher-level cognition and action modules.

**Challenges the pipeline must address:**

1. **Door and window detection** — reliably detect and localize doors/windows in the robot's field of view, determining position and extent in both the 2D image plane and 3D space, enabling approach trajectory planning.
2. **Door handle detection** — once a door is detected, identify handle location and type (lever vs. round knob), since type determines manipulation strategy.
3. **Classification of door properties** — opening direction (inward/outward), glass doors (transparency/reflection challenges), sliding doors vs. standard doors. Each category needs a different interaction strategy and must be reliably distinguished.
4. **Confidence estimation** — every detection/classification output gets a confidence score, so downstream modules can decide whether to act, request more sensor data, or flag for human intervention. Especially important under low light, occlusion, or sensor noise.

### Deliverables

1. A report detailing the approach and results.
2. A code repository with the experimental code.
3. A video demonstration of the results.

---

## Practical Information

| Field | Detail |
|---|---|
| Type of assignment | HBO bachelor internship (original framing) |
| Duration | 6 months (original framing) |
| Required skills | Python, 3D Computer Vision, Machine Learning (PyTorch and/or TensorFlow), ROS2 |

Note: this is now being run as an independent personal project — the originating group is no longer involved.
