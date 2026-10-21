# Detection accuracy and production requirements

## Current safeguards

- Image results use cross-model box suppression and a separate weapon verifier.
- Video results are grouped into object tracks by label and bounding-box overlap. Duplicate boxes in a sampled frame cannot inflate a track's observation count.
- A video weapon event is reported as verified only when the frame-level verifier marked a detection verified. Temporal persistence by itself is insufficient.
- Candidate events remain visible for human review. An empty weapon result is stated as “no verified weapon detections in sampled frames,” not proof that no weapon exists.
- These changes reduce known false-positive pathways. They do not establish a measured accuracy percentage or guarantee detection of every weapon.

## Requirements to establish and improve accuracy

1. **Representative, permissioned data:** collect legally authorized images and video across camera types, lighting, distance, occlusion, motion blur, weapon types, and visually similar non-weapons. Keep separate train, validation, and test sets by scene/camera, not adjacent frames.
2. **Ground-truth review:** have at least two qualified reviewers label boxes and weapon/non-weapon status, resolve disagreements, and retain provenance and permitted-use records.
3. **Metrics by class and condition:** publish precision, recall, false positives per image/minute, mAP at documented IoU thresholds, and confidence intervals. Report weapon subtypes and difficult conditions separately; do not use one overall score as a deployment claim.
4. **Threshold calibration:** tune confidence, NMS, weapon verification, temporal persistence, and tracking IoU against a held-out validation set. Freeze them for final testing; record model hashes, dataset version, and configuration.
5. **Temporal evaluation:** test clips with known positive/negative segments, occlusion, camera cuts, missed frames, and multiple objects. Measure event-level precision/recall and false alarms per hour, not only frame-level accuracy.
6. **Human review and audit:** require investigator confirmation before findings are treated as factual. Preserve original media, timestamps, model/configuration version, raw and filtered results, reviewer decisions, and an append-only audit history.
7. **Operational controls:** enforce upload size/type limits, access control, encryption at rest and in transit, retention/deletion rules, backups, error monitoring, and a documented incident process. Validate jurisdictional privacy and evidence-handling requirements before real investigations.
8. **Release gates:** block release on regressions in false-positive rate, calibration, robustness, security, or supported hardware performance. Test on the exact model weights and deployment environment intended for use.

## Antigravity implementation prompt

Paste this prompt into Antigravity with the repository open:

```text
You are improving the existing Crime-Investigation-AI Python/Streamlit repository. First inspect README.md, docs/ARCHITECTURE.md, config/settings.py, models/yolo_detector.py, models/weapon_verifier.py, models/video_processor.py, models/batch_processor.py, pages/investigation.py, and their existing tests. Preserve existing interfaces and the conservative rule that an unverified candidate is never presented as a confirmed weapon.

Goal: improve image/video crime-element detection quality, eliminate duplicate detections and false weapon claims, and make the investigation UI clear, professional, accessible, and auditable. Do not claim 100% accuracy or invent metrics.

Implement in small, reviewable changes:
1. Audit detector model names/weights and class mappings; ensure unsupported classes are not relabeled as weapons. Surface the actual model/weight version in results.
2. Make image-level cross-model deduplication class-aware and preserve distinct nearby objects. Add configurable, validated thresholds with documented defaults.
3. Make video object tracking account for label, bounding-box overlap, timestamps, and one observation per object per sampled frame. Do not verify weapons from persistence alone; require frame-level verifier status and report candidate/rejected/verified separately.
4. Ensure severity, summaries, reports, and exports consume verified weapon findings only; candidates must be explicitly marked as needing human review. A negative result means “not detected in analyzed frames,” never “absent.”
5. Improve the Streamlit evidence workspace with clear processing state, model/configuration metadata, per-event timestamps, confidence/status, readable empty states, and accessible keyboard/focus/contrast behavior. Keep evidence traceability and review actions.
6. Add focused regression tests for overlapping duplicate boxes, nearby distinct objects, duplicate detections in one frame, no-weapon clips, persistent false candidates, confirmed temporal tracks, camera cuts, and empty/failed input. Run the repository's relevant tests and report exact results.
7. Do not download or substitute model weights without recording their source, license, hash, and class map. Do not tune thresholds on the final held-out test set. Do not claim accuracy improvements until measured on a permissioned, independently labeled dataset.

Before editing, summarize the relevant architecture and likely risks. After editing, report changed files, exact checks run, limitations, and the dataset/labeling/validation work still needed to quantify accuracy. Keep all AI-generated interpretations clearly labeled as suggestions for human review.
```

