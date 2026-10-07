"""Service layer: orchestrates models / utilities into use-case APIs.

Services are kept Streamlit-free and I/O-light so they can be unit
tested without a server. The services added in this milestone are:

- `chat_assistant`:  a small retrieval-augmented chat over the
  latest EvidenceAnalysis (FLAN-T5 with deterministic fallback).
- `crime_timeline`:  builds an ordered Timeline from an analysis.
- `crime_prediction`: deterministic category risk distribution.
"""