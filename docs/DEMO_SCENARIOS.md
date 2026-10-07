# Gold Standard Demo Scenarios: Crime Investigation AI

This document provides a step-by-step guide to demonstrating the full power of the system to an examiner. The goal is to move from a single image analysis to a global criminal intelligence view.

## Scenario 1: The "First Contact" (Basic Workflow)
**Goal**: Demonstrate the core detection $\rightarrow$ summary $\rightarrow$ report pipeline.
1. **Action**: Upload a clear image/video of a crime scene (e.g., `robbery_scene_01.mp4`).
2. **Observe**:
    - YOLOv8 identifies suspects and weapons.
    - AI Summary generates a factual narrative.
    - A professional PDF report is generated.
3. **Narrative**: *"The system automatically processes raw evidence, extracts key forensic objects, and synthesizes a narrative for the investigator."*

## Scenario 2: The "Critical Audit" (Human-in-the-Loop)
**Goal**: Demonstrate that human expertise overrides AI.
1. **Action**: Go to **Human Review**. Find a "Candidate" weapon detection.
2. **Decision**: `REJECT` a false positive (e.g., a phone that looks like a gun).
3. **Observe**: 
    - The severity score drops.
    - The summary is marked as **OUTDATED**.
    - Regenerate the report $\rightarrow$ the "Human Verification Summary" now shows 1 Rejected detection.
4. **Narrative**: *"We ensure forensic integrity. AI suggests, but the human investigator decides. The system tracks these audits and invalidates stale reports automatically."*

## Scenario 3: The "Pattern Discovery" (Intelligence Layer)
**Goal**: Demonstrate Cross-Case Linking and Crime Series.
1. **Setup**: Upload 3-4 cases. Ensure Case A and Case B both contain a "Red SUV". Ensure Case B and Case C both contain a "Black Backpack".
2. **Action**: Navigate to **Link Analysis**.
3. **Observe**:
    - **Global Distribution**: Show the total count of Red SUVs across the city.
    - **Visual Link Graph**: Show Case A, B, and C connected in a cluster.
    - **Crime Series**: Point out that the AI has automatically grouped these into "Series #1".
4. **Narrative**: *"The system transforms from a case-processor to an intelligence platform. It identifies hidden connections—like a shared vehicle—linking seemingly unrelated crimes into a single criminal series."*

## Scenario 4: The "Big Picture" (Temporal & Spatial)
**Goal**: Demonstrate the intelligence landscape.
1. **Action**: Use the "Set Case Location" tool to assign different city coordinates to the cases.
2. **Observe**:
    - **Spatial Map**: See the "Hotspots" of criminal activity.
    - **Temporal Trends**: Show the line chart of evidence frequency over time (e.g., a spike in weapon detections in October).
3. **Action**: Click **Export Intelligence Report**.
4. **Narrative**: *"Finally, we can visualize the 'Where' and 'When'. We can see the geographic spread of the series and the temporal evolution of the crime wave, all exportable into a formal intelligence briefing."*

## Summary of "Wow" Factors for the Examiner:
- **Local AI**: "No data leaves this machine; total privacy."
- **State Management**: "The 'Outdated' flag prevents forensic errors."
- **Community Detection**: "We use the Louvain algorithm to find criminal gangs automatically."
- **End-to-End**: "From a raw MP4 to a professional Intelligence PDF."
