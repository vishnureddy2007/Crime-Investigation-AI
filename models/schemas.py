"""
Core data schemas for the Crime Investigation AI.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Optional


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class Detection:
    class_name: str
    label: str
    confidence: float
    bbox: BoundingBox
    source: str = "general"
    weapon_status: str = ""  # "verified" | "candidate" | ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "class_name": self.class_name,
            "label": self.label,
            "confidence": self.confidence,
            "bbox": {"x1": self.bbox.x1, "y1": self.bbox.y1, "x2": self.bbox.x2, "y2": self.bbox.y2},
            "source": self.source,
            "weapon_status": self.weapon_status,
        }


@dataclass
class DetectionResult:
    source_name: str
    timestamp: datetime
    detections: list[Detection] = field(default_factory=list)
    raw_count: int = 0
    annotated_image: Any | None = None
    model_name: str = "yolov8n"
    source_tag: str = "general"
    models_used: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.detections)

    def classes(self) -> list[str]:
        return [d.class_name for d in self.detections]

    def labels(self) -> list[str]:
        return [d.label for d in self.detections]

    def counts_by_label(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for d in self.detections:
            counts[d.label] = counts.get(d.label, 0) + 1
        return counts

    def average_confidence(self) -> float:
        if not self.detections:
            return 0.0
        return sum(d.confidence for d in self.detections) / len(self.detections)


@dataclass
class EvidenceAnalysis:
    source_name: str
    source_type: str
    counts_by_label: dict[str, int]
    total_objects: int
    unique_labels: list[str]
    average_confidence: float
    person_count: int
    verified_weapon_count: int
    candidate_weapon_count: int
    vehicle_count: int
    bag_count: int
    severity_score: int
    severity_level: str
    suggested_category: str
    key_observations: list[str]
    has_threat: bool
    weapon_count: int
    frame_count: int
    timestamp: datetime = field(default_factory=datetime.now)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "source_type": self.source_type,
            "counts_by_label": self.counts_by_label,
            "total_objects": self.total_objects,
            "unique_labels": self.unique_labels,
            "average_confidence": self.average_confidence,
            "person_count": self.person_count,
            "verified_weapon_count": self.verified_weapon_count,
            "candidate_weapon_count": self.candidate_weapon_count,
            "vehicle_count": self.vehicle_count,
            "bag_count": self.bag_count,
            "severity_score": self.severity_score,
            "severity_level": self.severity_level,
            "suggested_category": self.suggested_category,
            "key_observations": self.key_observations,
            "has_threat": self.has_threat,
            "weapon_count": self.weapon_count,
            "frame_count": self.frame_count,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DetailedNarrativeSummary:
    case_id: str
    case_overview: str
    evidence_reviewed: str
    chronological_events: str
    detected_objects: str
    verified_findings: str
    possible_findings: str
    rejected_findings: str
    potential_crime_activity: str
    important_evidence: str
    uncertainties: str
    investigation_summary: str = ""
    summary_id: str = "sum_001"

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "case_overview": self.case_overview,
            "evidence_reviewed": self.evidence_reviewed,
            "chronological_events": self.chronological_events,
            "detected_objects": self.detected_objects,
            "verified_findings": self.verified_findings,
            "possible_findings": self.possible_findings,
            "rejected_findings": self.rejected_findings,
            "potential_crime_activity": self.potential_crime_activity,
            "important_evidence": self.important_evidence,
            "uncertainties": self.uncertainties,
            "investigation_summary": self.investigation_summary,
            "summary_id": self.summary_id,
        }


@dataclass
class CrimeSituationAnalysis:
    likely_activity_pattern: str
    possible_sequence_of_events: str
    potential_next_activity: str
    suspicious_behavior_indicators: list[str]
    risk_indicators: list[str]
    supporting_evidence: list[str]
    confidence_level: str
    uncertainties: str
    alternative_explanations: str

    def as_dict(self):
        return self.__dict__


@dataclass
class StoryboardEvent:
    """An animated event in the 3D scene."""
    timestamp: float  # Seconds from start
    actor_id: str      # ID of the object/person
    action: str        # MOVE_TO, APPEAR, DISAPPEAR, HIGHLIGHT
    target_pos: list[float] # [x, y, z]
    label: str         # "Person A", "Weapon #1"
    status: str        # VERIFIED, PREDICTED
    description: str   # Caption for the video


@dataclass
class StoryboardPlan:
    """The structured plan used by Blender to render the video."""
    case_id: str
    environment_bounds: list[float] # [x, y, z]
    objects: list[dict] # List of objects with initial_pos, kind, label, verified status
    timeline: list[StoryboardEvent]
    camera_shots: list[dict] # List of {pos, target, duration, type}
    total_duration: float


# =====================================================================
# Milestone 3: FrameResult + VideoAnalysisResult
# =====================================================================
@dataclass
class FrameResult:
    """One extracted video frame + its detection."""
    index: int
    timestamp_sec: float
    detection: DetectionResult | None
    image_path: Optional[Any] = None
    annotated_path: Optional[Any] = None

    @property
    def detection_count(self) -> int:
        if self.detection is None:
            return 0
        return self.detection.count

    @property
    def average_confidence(self) -> float:
        if self.detection is None:
            return 0.0
        return self.detection.average_confidence()

    @property
    def evidence_score(self) -> float:
        """Score used for keyframe ranking: count * avg_confidence."""
        return self.detection_count * self.average_confidence


@dataclass
class VideoAnalysisResult:
    """All FrameResults + metadata + keyframe indices."""
    source_name: str
    timestamp: datetime
    frame_results: list[FrameResult]
    metadata: dict[str, Any] = field(default_factory=dict)
    keyframe_indices: list[int] = field(default_factory=list)
    model_name: str = "unknown"

    @property
    def frame_count(self) -> int:
        return len(self.frame_results)

    @property
    def keyframes(self) -> list[FrameResult]:
        return [self.frame_results[i] for i in self.keyframe_indices if 0 <= i < len(self.frame_results)]

    def total_detections(self) -> int:
        return sum(fr.detection_count for fr in self.frame_results)

    def aggregate_counts_by_label(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for fr in self.frame_results:
            if fr.detection is None:
                continue
            for d in fr.detection.detections:
                counts[d.label] = counts.get(d.label, 0) + 1
        return counts


# =====================================================================
# Milestone 5+: InvestigationSummary
# =====================================================================
@dataclass
class InvestigationSummary:
    """Container for both template and AI-generated summary text."""
    source_name: str
    source_type: str
    template_text: str
    ai_text: str | None = None
    used_ai: bool = False
    model_name: str = "template"
    generation_time_sec: float = 0.0
    evidence_snapshot: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def primary_text(self) -> str:
        """Returns AI text if available, otherwise template text."""
        return self.ai_text if (self.used_ai and self.ai_text) else self.template_text

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "source_type": self.source_type,
            "template_text": self.template_text,
            "ai_text": self.ai_text,
            "used_ai": self.used_ai,
            "model_name": self.model_name,
            "generation_time_sec": self.generation_time_sec,
            "evidence_snapshot": self.evidence_snapshot,
            "timestamp": self.timestamp.isoformat(),
        }


# =====================================================================
# Milestone 6+: ReportData
# =====================================================================
@dataclass
class ReportData:
    """Everything a report renderer needs."""
    title: str
    report_id: str
    generated_at: datetime
    source_name: str
    source_type: str
    model_name: str
    summary_text: str
    severity_score: int
    severity_level: str
    suggested_category: str
    has_threat: bool
    counts_by_label: dict[str, int]
    total_objects: int
    person_count: int
    weapon_count: int
    vehicle_count: int
    bag_count: int
    key_observations: list[str]
    average_confidence: float
    remarks: str
    app_version: str
    human_total_detections: int = 0
    human_confirmed_count: int = 0
    human_rejected_count: int = 0
    human_pending_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "report_id": self.report_id,
            "generated_at": self.generated_at.isoformat(),
            "source_name": self.source_name,
            "source_type": self.source_type,
            "model_name": self.model_name,
            "summary_text": self.summary_text,
            "severity_score": self.severity_score,
            "severity_level": self.severity_level,
            "suggested_category": self.suggested_category,
            "has_threat": self.has_threat,
            "counts_by_label": self.counts_by_label,
            "total_objects": self.total_objects,
            "person_count": self.person_count,
            "weapon_count": self.weapon_count,
            "vehicle_count": self.vehicle_count,
            "bag_count": self.bag_count,
            "key_observations": self.key_observations,
            "average_confidence": self.average_confidence,
            "remarks": self.remarks,
            "app_version": self.app_version,
            "human_total_detections": self.human_total_detections,
            "human_confirmed_count": self.human_confirmed_count,
            "human_rejected_count": self.human_rejected_count,
            "human_pending_count": self.human_pending_count,
        }


# =====================================================================
# Phase 13+: IntelligenceReportData (Cross-Case Analytics)
# =====================================================================
@dataclass
class IntelligenceReportData:
    """Data for cross-case intelligence report."""
    report_id: str
    generated_at: datetime
    global_distribution: dict[str, int]
    crime_series: dict[int, list[int]]
    case_details: dict[int, str]
    app_version: str


# =====================================================================
# Phase 14+: StoryboardScene + Storyboard
# =====================================================================
@dataclass
class StoryboardScene:
    """One captioned panel in the storyboard."""
    index: int
    title: str
    caption: str
    image: Any  # PIL Image or None
    duration_sec: float
    based_on_real_frame: bool


@dataclass
class Storyboard:
    """Ordered list of storyboard scenes."""
    source_name: str
    source_type: str
    scenes: list[StoryboardScene]
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def scene_count(self) -> int:
        return len(self.scenes)

    @property
    def total_duration_sec(self) -> float:
        return sum(s.duration_sec for s in self.scenes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "source_type": self.source_type,
            "scenes": [
                {
                    "index": s.index,
                    "title": s.title,
                    "caption": s.caption,
                    "duration_sec": s.duration_sec,
                    "based_on_real_frame": s.based_on_real_frame,
                    # image is not serialized
                }
                for s in self.scenes
            ],
            "timestamp": self.timestamp.isoformat(),
        }


# =====================================================================
# AnalysisInput - adapter for image/video analysis input
# =====================================================================
@dataclass
class AnalysisInput:
    """Adapter for either image or video analysis input."""
    source_name: str
    counts_by_label: dict[str, int]
    average_confidence: float
    source_type: str = "image"
    frame_count: int = 1

    @classmethod
    def from_image(cls, detection_result: DetectionResult) -> "AnalysisInput":
        """Create AnalysisInput from an image DetectionResult."""
        return cls(
            source_name=detection_result.source_name,
            counts_by_label=detection_result.counts_by_label(),
            average_confidence=detection_result.average_confidence(),
            source_type="image",
            frame_count=1,
        )

    @classmethod
    def from_video(cls, video_result: VideoAnalysisResult) -> "AnalysisInput":
        """Create AnalysisInput from a VideoAnalysisResult."""
        # Calculate overall average confidence across all frames with detections
        total_conf = 0.0
        total_dets = 0
        for fr in video_result.frame_results:
            if fr.detection and fr.detection.detections:
                total_conf += sum(d.confidence for d in fr.detection.detections)
                total_dets += len(fr.detection.detections)
        avg_conf = total_conf / total_dets if total_dets > 0 else 0.0
        
        return cls(
            source_name=video_result.source_name,
            counts_by_label=video_result.aggregate_counts_by_label(),
            average_confidence=avg_conf,
            source_type="video",
            frame_count=video_result.frame_count,
        )


# =====================================================================
# StructuredSceneNarrative - structured scene plan for 3D reconstruction
# =====================================================================
@dataclass
class StructuredSceneNarrative:
    """Structured scene narrative for 3D animation planning."""
    environment: str
    events: list[dict[str, Any]]
    objects: list[dict[str, Any]]
    people_count: int | None = None
    camera_plan: dict[str, Any] = field(default_factory=dict)
    visualization_notes: list[str] = field(default_factory=list)