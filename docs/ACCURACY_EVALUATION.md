# Forensic Detection Accuracy Evaluation Protocol

This document establishes the scientific evaluation and benchmarking methodology for the Crime-Investigation-AI detection, tracking, and weapon verification pipeline.

> [!IMPORTANT]
> **Zero-Claim Policy**: No claim of accuracy improvement or operational readiness may be published until evaluated on an independent, permissioned held-out benchmark following this protocol. Do not invent metrics or claim 100% accuracy under any circumstances.

---

## 1. Permissioned Dataset Protocol

### 1.1 Sourcing & Legal Permission
- **Representative Media**: Data must represent real operational surveillance conditions:
  - Fixed CCTV cameras (overhead angles, variable focal lengths, 10–30 fps).
  - Body-worn cameras (high motion blur, rapid panning, variable exposure).
  - Mobile/handheld recordings (varying aspect ratios, compression artifacts).
- **Consent & Authorization**: All footage must have explicit legal authorization, chain of custody logs, and GDPR/privacy compliance (redacting uninvolved bystanders where legally required).

### 1.2 Non-Leaking Scene/Camera Splits
- **Strict Scene Separation**: Splitting by random frame assignment leaks adjacent, visually identical frames across splits and invalidates results.
- **Split Rule**: Data must be partitioned strictly by **unique physical scene** or **unique camera ID**:
  - `Train Set` (60%): Unique scenes/cameras.
  - `Validation Set` (20%): Separate scenes/cameras; used exclusively for calibrating thresholds (e.g., NMS IoU, weapon scan threshold, verification confidence).
  - `Held-Out Test Set` (20%): Completely distinct scenes/cameras never observed during training or tuning. **Frozen for final benchmark only.**

---

## 2. Multi-Reviewer Ground Truth Labeling

### 2.1 Annotation Standards
- Every frame must be labeled by at least **two qualified forensic reviewers** working independently.
- Annotation requirements:
  - Tight 2D bounding box `[x1, y1, x2, y2]`.
  - Class label (`person`, `vehicle`, `bag`, `knife`, `firearm_handgun`, `firearm_rifle`, `non_weapon_distractor`).
  - Difficult condition flags (`low_light`, `motion_blur`, `occluded_gt_50`, `small_object_lt_32px`).
  - Track ID linking physical objects across temporal video segments.

### 2.2 Disagreement Resolution
- Compute inter-annotator agreement using **Cohen’s Kappa ($\kappa$)** or **Fleiss’ Kappa** for class labeling and IoU overlap ($\ge 0.70$) for bounding box agreement.
- Instances with reviewer disagreement ($\kappa < 0.85$ or IoU $< 0.70$) must be adjudicated by a senior forensic lead or discarded from the benchmark.

---

## 3. Evaluation Metrics

### 3.1 Frame-Level Metrics
For each class $c$ (especially weapon categories: knife, handgun, rifle, etc.):
1. **Precision ($P_c$)**:
   $$\text{Precision} = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c}$$
2. **Recall ($R_c$)**:
   $$\text{Recall} = \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c}$$
3. **Mean Average Precision (mAP)**:
   - Reported at $\text{mAP}@0.50$ and $\text{mAP}@0.50:0.95$.
4. **False Alarm Rate**:
   - Number of False Positives per Image (FPPI) or per 1,000 background frames.

### 3.2 Video Event-Level Metrics
Because single-frame metrics do not capture temporal reliability:
1. **Event Precision & Recall**:
   - An event is a True Positive if an overlapping ground-truth weapon track matches with temporal IoU $\ge 0.50$ and label match.
2. **False Alarms per Video Minute / Hour (FA/hr)**:
   $$\text{FA/hr} = \frac{\text{Total False Positive Weapon Events}}{\text{Total Video Duration in Hours}}$$
3. **Mean Time to Detect (MTTD)**:
   - Average duration in seconds between a weapon becoming un-occluded and the system issuing a candidate alert.

### 3.3 Condition-Stratified Reporting
Metrics must be reported broken down by difficulty conditions:
- **Lighting**: Day / Well-lit vs. Low-light / Night-vision IR.
- **Occlusion**: Clear ($\le 20\%$) vs. Partial ($21\% - 50\%$) vs. Heavy ($> 50\%$).
- **Object Resolution**: Large ($> 96\times 96$ px) vs. Medium ($32\times 32$ to $96\times 96$ px) vs. Small ($< 32\times 32$ px).
- **Distractor Classes**: Specific false positive rate on visually similar everyday objects:
  - Power tools, drills, hammers.
  - Smart phones, remote controls.
  - Thermal flasks, water bottles, umbrellas.

---

## 4. Statistical Rigor & Reporting Checklist

When publishing or logging benchmark results, every report must include:
1. **Model Weights Provenance**:
   - Checksum: SHA-256 hash of all active model weights files (`yolov8n.pt`, `models/weapon.pt`, `models/threat_weapon.pt`).
   - Architecture & source license (e.g. AGPL-3.0 for YOLOv8).
2. **Frozen Configuration**:
   - Confidence threshold (e.g. `0.25`).
   - NMS IoU threshold (e.g. `0.45`).
   - Weapon scan threshold (e.g. `0.15`).
   - Weapon verify threshold (e.g. `0.45`).
   - Tracking temporal gap (e.g. `2.0s`).
3. **Confidence Intervals**:
   - 95% bootstrap confidence intervals (1,000 resamples) for Precision, Recall, and FA/hr.
4. **Caveat Standard**:
   - Explicit disclaimer that AI outputs are investigative suggestions requiring human forensic review, and that negative findings indicate non-detection in analyzed footage rather than affirmative proof of absence.
