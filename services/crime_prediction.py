"""Deterministic crime-category risk prediction.

`predict_risk(analysis)` returns a `RiskAssessment` — a probability
distribution over the project's crime categories. The model is a
hand-tuned, transparent weighted combination of evidence signals:

  - weapon_count   raises assault & robbery
  - bag_count      raises theft
  - vehicle_count  raises vehicle_incident
  - person_count   slightly raises every "people present" category
  - high_confidence raises everything proportionally
  - low evidence   concentrates mass on "suspicious_activity"

The result is always renormalised so the values sum to 1.0 (within
1e-6). This is deliberately not an ML model — it's a rule-based
heuristic that is testable, deterministic, and never throws.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from models.schemas import EvidenceAnalysis


# The full category set. Keep in sync with `CATEGORY_RULES` keys.
CATEGORIES: tuple[str, ...] = (
    "assault",
    "robbery",
    "theft",
    "vehicle_incident",
    "suspicious_activity",
    "unknown",
)


@dataclass(frozen=True)
class RiskAssessment:
    """Probability distribution over the supported crime categories."""

    source_name: str
    scores: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        # Round to 4 decimal places for stable JSON.
        return {
            "source_name": self.source_name,
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
        }

    @property
    def top_category(self) -> str:
        return max(self.scores.items(), key=lambda kv: kv[1])[0]

    @property
    def top_score(self) -> float:
        return self.scores[self.top_category]


def predict_risk(analysis: EvidenceAnalysis | None) -> RiskAssessment:
    """Return a risk distribution for `analysis`, or an empty one."""
    if analysis is None:
        return RiskAssessment(source_name="", scores={c: 0.0 for c in CATEGORIES})

    # Empty evidence → suspicious_activity dominates from the start.
    # We check the *component* counts, not total_objects, because callers
    # sometimes pass partially-populated analyses.
    any_evidence = any([
        analysis.weapon_count,
        analysis.person_count,
        analysis.vehicle_count,
        analysis.bag_count,
        analysis.total_objects,
    ])
    if not any_evidence:
        scores = {c: 0.025 for c in CATEGORIES}
        scores["suspicious_activity"] = 0.85
        # Renormalise so values sum to exactly 1.0.
        total = sum(scores.values())
        scores = {k: v / total for k, v in scores.items()}
        return RiskAssessment(source_name=analysis.source_name, scores=scores)

    # Start from a low-uniform baseline so every category has a chance.
    scores = {c: 0.10 for c in CATEGORIES}

    # Weapons escalate assault / robbery the most. Tuned so that a
    # single weapon pushes assault/robbery above 0.20 after
    # renormalisation across the 6 baseline categories.
    if analysis.weapon_count > 0:
        scores["assault"]            += 1.50 + 0.30 * (analysis.weapon_count - 1)
        scores["robbery"]            += 1.20 + 0.20 * (analysis.weapon_count - 1)
        scores["suspicious_activity"] += 0.10

    # Bags push theft.
    if analysis.bag_count > 0:
        scores["theft"]              += 1.80 + 0.10 * (analysis.bag_count - 1)

    # Vehicles push vehicle_incident.
    if analysis.vehicle_count > 0:
        scores["vehicle_incident"]   += 2.00 + 0.15 * (analysis.vehicle_count - 1)

    # Persons bump every people-relevant category slightly.
    if analysis.person_count > 0:
        bump = 0.20 * min(analysis.person_count, 5)
        scores["assault"]            += bump
        scores["robbery"]            += bump
        scores["suspicious_activity"] += bump

    # High-confidence detections lift everything proportionally.
    if analysis.average_confidence >= 0.85:
        for k in scores:
            scores[k] *= 1.15

    # Renormalise to sum to 1.0.
    total = sum(scores.values())
    if total > 0:
        scores = {k: v / total for k, v in scores.items()}

    # Floor to 0 so the dataclass stays honest.
    scores = {k: max(0.0, v) for k, v in scores.items()}

    return RiskAssessment(source_name=analysis.source_name, scores=scores)