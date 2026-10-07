# Thesis Technical Content: AI Crime Investigation Assistant

This document provides structured technical descriptions of the system's core innovations, intended for use in a final-year B.Tech project thesis or academic report.

---

## 1. System Architecture Overview
The system follows a modular, layered architecture designed for high reliability and data privacy.

- **Presentation Layer (Streamlit)**: An interactive web-based dashboard that handles the forensic workflow.
- **Orchestration Layer (Services)**: Coordinates complex tasks such as the Crime Timeline synthesis, Risk Prediction, and Chat Assistant.
- **Model Layer (YOLOv8 & Qwen3)**: 
    - **YOLOv8**: Performs multi-source object detection (Images/Video) to identify forensic-relevant classes.
    - **Qwen3 14B (Local AI)**: A Large Language Model hosted via Ollama that transforms structured detection logs into human-readable narratives.
- **Persistence Layer (SQLite)**: A relational database that stores cases, analyses, summaries, and reports with full state management.

---

## 2. Human-in-the-Loop (HITL) Verification Logic
A critical contribution of this project is the **Human-in-the-Loop** system, which ensures that AI detections are audited by a qualified investigator before being finalized.

### Logic Flow:
1. **Candidate Generation**: YOLOv8 detects objects. Detections are marked as `verified` (above threshold) or `candidate` (below threshold).
2. **Human Audit**: The investigator reviews each candidate in the **Human Review** module.
3. **Decision Mapping**:
    - `CONFIRM` $\rightarrow$ Object is promoted to **Verified**.
    - `REJECT` $\rightarrow$ Object is marked as **False Positive** and removed from the final evidence count.
    - `UNCERTAIN` $\rightarrow$ Object remains a candidate for further investigation.
4. **Downstream Propagation**: The Human Review decision overrides the AI's original finding. The final **Forensic Report** and **3D Reconstruction** are built using the *human-verified* counts, not the raw AI counts.

---

## 3. AI-Driven Narrative-to-Video Pipeline
The system implements a novel approach to crime scene visualization by using LLM-driven scene planning.

### The Process:
1. **Summary Generation**: Qwen3 14B generates a concise, factual summary of the evidence.
2. **Narrative Decomposition**: The summary is passed back to the AI with a specific prompt to decompose the story into a sequence of **Visual Beats** (scenes).
3. **Storyboard Mapping**: Each visual beat is mapped to a real evidence frame (if available) or a professional placeholder.
4. **Temporal Synthesis**: Using OpenCV's `VideoWriter`, these scenes are rendered into an MP4 video with cross-fade transitions, creating a "probable reconstruction" of the incident.

---

## 4. State Management & "Outdated" Artifacts
To maintain forensic integrity, the system prevents investigators from relying on stale data.

- **Invalidation Trigger**: Whenever the underlying evidence is updated (e.g., a new analysis run or a human review decision), a `mark_case_outdated()` trigger is fired.
- **Artifact Tracking**: Summaries, Reports, and Storyboards are marked with an `is_outdated` flag.
- **UI Guardrails**: The interface surfaces a prominent warning whenever an outdated artifact is accessed, forcing the investigator to regenerate the report to reflect the current "truth."

---

## 5. Performance & Robustness Metrics
- **Local AI Privacy**: 100% of the AI processing happens on the host machine (no cloud calls).
- **Deterministic Fallback**: In the event of an LLM failure, the system falls back to a template-based generator, ensuring 100% availability.
- **Test Coverage**: The system is validated by a comprehensive suite of over 500 tests, including robustness tests for empty evidence and service outages.
