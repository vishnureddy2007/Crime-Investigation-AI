"""
Combined-evidence analysis helpers.

`build_combined_analysis` walks a list of `AnalysisInput` (one per
file), aggregates their counts, and produces a single
`EvidenceAnalysis` that the report generator and reconstruction
module consume down the line.

It is deliberately a function (not a class) so it's trivial to
unit-test and so the input list can be assembled anywhere — from
the batch processor, from a saved case, or from a manual replay.

Phase 46: weapons are split into *verified* and *candidate* buckets.
Only verified weapons drive severity / `has_threat` / category.
A bottle in the image is NEVER a weapon.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable

from config import CATEGORY_RULES
from models.evidence_analyzer import (
    _build_observations,
    _compute_severity,
    _severity_level,
    _suggest_category,
)
from models.schemas import AnalysisInput, EvidenceAnalysis


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def build_combined_analysis(
    inputs: Iterable[AnalysisInput],
    total_files: int,
) -> EvidenceAnalysis:
    """
    Merge a list of per-file `AnalysisInput` into one `EvidenceAnalysis`.

    The combined analysis is the union of every per-file label
    counts, the average of the per-file confidences, and the sum
    of person/verified_weapon/candidate_weapon/vehicle/bag counts.
    `frame_count` is the sum of every file's frame count (1 for
    images, N for videos).
    """
    inputs = list(inputs)
    if not inputs:
        return EvidenceAnalysis(
            source_name="combined",
            source_type="batch",
            counts_by_label={},
            total_objects=0,
            unique_labels=[],
            average_confidence=0.0,
            person_count=0,
            verified_weapon_count=0,
            candidate_weapon_count=0,
            weapon_count=0,
            vehicle_count=0,
            bag_count=0,
            severity_score=0,
            severity_level="low",
            suggested_category="unknown",
            key_observations=["No evidence provided."],
            has_threat=False,
            frame_count=0,
            timestamp=datetime.now(),
        )

    combined_counts: dict[str, int] = {}
    confs: list[float] = []
    total_frames = 0
    person_count = 0
    verified_weapon_count = 0
    candidate_weapon_count = 0
    knife_count = 0
    bottle_count = 0
    vehicle_count = 0
    bag_count = 0
    file_names: list[str] = []

    for inp in inputs:
        file_names.append(inp.source_name)
        total_frames += inp.frame_count
        confs.append(float(inp.average_confidence))

        for label, n in inp.counts_by_label.items():
            combined_counts[label] = combined_counts.get(label, 0) + int(n)

        # Phase 46: weapons live in two buckets — `weapon`/`knife`
        # (verified) and `candidate_weapon` (low-confidence hint).
        # Bottle is NOT a weapon under any circumstances.
        p = combined_counts.get("person", 0)
        k = combined_counts.get("knife", 0)
        b = combined_counts.get("bottle", 0)
        w = combined_counts.get("weapon", 0)
        cw = combined_counts.get("candidate_weapon", 0)
        v = combined_counts.get("vehicle", 0)
        bag = combined_counts.get("bag", 0)
        person_count = p
        knife_count = k
        bottle_count = b
        verified_weapon_count = k + w
        candidate_weapon_count = cw
        vehicle_count = v
        bag_count = bag

    total_objects = sum(combined_counts.values())
    unique_labels = sorted(label for label, n in combined_counts.items() if n > 0)
    avg_conf = sum(confs) / len(confs) if confs else 0.0

    # --- Severity / category -------------------------------------------
    severity = _compute_severity(
        verified_weapon_count=verified_weapon_count,
        candidate_weapon_count=candidate_weapon_count,
        person_count=person_count,
        vehicle_count=vehicle_count,
        bag_count=bag_count,
        avg_conf=avg_conf,
    )
    severity_level = _severity_level(severity)

    present_labels = set(unique_labels)
    if verified_weapon_count > 0:
        present_labels.add("verified_weapon")
    suggested = _suggest_category(present_labels)

    observations = _build_observations(
        person_count=person_count,
        verified_weapon_count=verified_weapon_count,
        candidate_weapon_count=candidate_weapon_count,
        knife_count=knife_count,
        bottle_count=bottle_count,
        vehicle_count=vehicle_count,
        bag_count=bag_count,
        total_objects=total_objects,
        source_type="batch",
        frame_count=total_frames,
    )
    # Prepend a clear batch-summary line so the report doesn't lose
    # the "files" context.
    summary_line = (
        f"Combined analysis of {total_files} file(s) "
        f"({len(inputs)} successfully processed, "
        f"{sum(1 for inp in inputs if inp.source_type == 'image')} image(s), "
        f"{sum(1 for inp in inputs if inp.source_type == 'video')} video(s))."
    )
    observations = [summary_line] + observations

    return EvidenceAnalysis(
        source_name=f"combined ({len(inputs)} file(s))",
        source_type="batch",
        counts_by_label=combined_counts,
        total_objects=total_objects,
        unique_labels=unique_labels,
        average_confidence=avg_conf,
        person_count=person_count,
        verified_weapon_count=verified_weapon_count,
        candidate_weapon_count=candidate_weapon_count,
        weapon_count=verified_weapon_count,
        vehicle_count=vehicle_count,
        bag_count=bag_count,
        severity_score=severity,
        severity_level=severity_level,
        suggested_category=suggested,
        key_observations=observations,
        has_threat=verified_weapon_count > 0,
        frame_count=total_frames,
        timestamp=datetime.now(),
    )


# ----------------------------------------------------------------------
# Helpers (kept here for backward-compat with any test that imported
# them from `models.analysis_engine` rather than `models.evidence_analyzer`).
# ----------------------------------------------------------------------
__all__ = ["build_combined_analysis"]
