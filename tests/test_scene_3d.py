"""Tests for Phase 49 — 3D scene plan pipeline (honest abstraction)."""
from __future__ import annotations

from datetime import datetime

import pytest

from models.scene_3d import (
    AI3DProvider,
    LocalProceduralProvider,
    Scene3DCamera,
    Scene3DGenerator,
    Scene3DObject,
    Scene3DPlan,
    Scene3DRenderResult,
)
from models.schemas import EvidenceAnalysis


def _analysis(**kw) -> EvidenceAnalysis:
    base = dict(
        source_name="clip.mp4",
        source_type="video",
        counts_by_label={"person": 2, "weapon": 1, "candidate_weapon": 1, "vehicle": 1},
        total_objects=5,
        unique_labels=["person", "weapon", "candidate_weapon", "vehicle"],
        average_confidence=0.8,
        person_count=2,
        verified_weapon_count=1,
        candidate_weapon_count=1,
        weapon_count=1,
        vehicle_count=1,
        bag_count=0,
        severity_score=70,
        severity_level="high",
        suggested_category="robbery",
        key_observations=[],
        has_threat=True,
        frame_count=10,
        timestamp=datetime.now(),
    )
    base.update(kw)
    return EvidenceAnalysis(**base)


# ----------------------------------------------------------------------
# Schema
# ----------------------------------------------------------------------
class TestScene3DPlanSchema:
    def test_plan_serialises_to_json(self) -> None:
        plan = Scene3DPlan(
            plan_id="p1",
            source_name="x.jpg",
            generated_at=datetime(2026, 1, 1, 12, 0, 0),
            scene_bounds_xyz=(4.0, 2.5, 4.0),
            objects=[
                Scene3DObject(
                    object_id="p1", kind="person", label="Person",
                    verified=True, position_xyz=(0.0, 0.0, 0.0),
                    size_xyz=(0.5, 1.7, 0.3),
                ),
            ],
            cameras=[Scene3DCamera(
                position_xyz=(0.0, 4.0, 4.0),
                look_at_xyz=(0.0, 0.0, 0.0),
            )],
        )
        s = plan.to_json()
        assert "person" in s
        assert plan.object_count == 1

    def test_object_counts_reflect_kind(self) -> None:
        plan = Scene3DPlan(
            plan_id="p1",
            source_name="x.jpg",
            generated_at=datetime.now(),
            scene_bounds_xyz=(4.0, 2.5, 4.0),
            objects=[
                Scene3DObject("w1", "weapon", "Gun", True, (0, 0, 0), (0.3, 0.05, 0.05)),
                Scene3DObject("w2", "weapon", "candidate", False, (0, 0, 0), (0.3, 0.05, 0.05)),
            ],
            cameras=[],
        )
        assert plan.verified_weapon_count == 1
        assert plan.candidate_weapon_count == 1


# ----------------------------------------------------------------------
# Generator
# ----------------------------------------------------------------------
class TestScene3DGenerator:
    def test_only_verified_weapons_become_objects(self) -> None:
        gen = Scene3DGenerator()
        plan = gen.build_plan(_analysis())
        weapon_objs = [o for o in plan.objects if o.kind == "weapon"]
        assert len(weapon_objs) == 1
        assert weapon_objs[0].verified is True
        # Candidate weapon is surfaced as a warning, NOT as a mesh.
        assert any("candidate" in w.lower() for w in plan.warnings)

    def test_no_weapons_in_plan_when_zero(self) -> None:
        a = _analysis(verified_weapon_count=0, candidate_weapon_count=0,
                      weapon_count=0, has_threat=False,
                      counts_by_label={"person": 1})
        plan = Scene3DGenerator().build_plan(a)
        assert all(o.kind != "weapon" for o in plan.objects)
        assert plan.verified_weapon_count == 0

    def test_person_count_matches_analysis(self) -> None:
        a = _analysis(person_count=3)
        plan = Scene3DGenerator().build_plan(a)
        persons = [o for o in plan.objects if o.kind == "person"]
        assert len(persons) == 3

    def test_vehicle_and_bag_counts_match(self) -> None:
        a = _analysis(vehicle_count=2, bag_count=2)
        plan = Scene3DGenerator().build_plan(a)
        assert sum(1 for o in plan.objects if o.kind == "vehicle") == 2
        assert sum(1 for o in plan.objects if o.kind == "bag") == 2

    def test_plan_id_includes_safe_filename(self) -> None:
        plan = Scene3DGenerator().build_plan(_analysis())
        assert "clip" in plan.plan_id

    def test_scene_bounds_grow_with_objects(self) -> None:
        small = Scene3DGenerator().build_plan(
            _analysis(person_count=0, verified_weapon_count=0,
                      vehicle_count=0, bag_count=0,
                      counts_by_label={}, total_objects=0),
        )
        big = Scene3DGenerator().build_plan(
            _analysis(person_count=10, vehicle_count=5),
        )
        assert big.scene_bounds_xyz[0] > small.scene_bounds_xyz[0]


# ----------------------------------------------------------------------
# Local provider (honest fallback — NEVER fake 3D)
# ----------------------------------------------------------------------
class TestLocalProceduralProvider:
    def test_local_provider_returns_fallback_status(self) -> None:
        provider = LocalProceduralProvider()
        plan = Scene3DGenerator().build_plan(_analysis())
        result = provider.render(plan)
        assert result.provider_status == "local-fallback"
        assert result.rendered_asset_url == ""
        assert result.thumbnail_data_url == ""
        assert result.provider_name == "local-procedural"
        assert "no external 3d provider configured" in result.note.lower() or \
               "no external 3d" in result.note.lower()

    def test_local_provider_does_not_invent_url(self) -> None:
        """The local provider must NEVER fabricate a URL that would
        look like a real 3D render."""
        provider = LocalProceduralProvider()
        plan = Scene3DGenerator().build_plan(_analysis())
        result = provider.render(plan)
        assert not result.rendered_asset_url
        assert not result.thumbnail_data_url

    def test_generate_returns_both_plan_and_result(self) -> None:
        gen = Scene3DGenerator(provider=LocalProceduralProvider())
        plan, result = gen.generate(_analysis())
        assert plan.object_count >= 1
        assert result.provider_status == "local-fallback"


# ----------------------------------------------------------------------
# Provider protocol compliance
# ----------------------------------------------------------------------
class _RecordingProvider:
    """Minimal provider used to confirm the protocol contract."""

    def __init__(self) -> None:
        self.name = "recording"
        self.received: list[Scene3DPlan] = []

    def render(self, plan: Scene3DPlan) -> Scene3DRenderResult:
        self.received.append(plan)
        return Scene3DRenderResult(
            provider_name=self.name,
            provider_status="rendered",
            rendered_asset_url="https://example.com/scene.glb",
            thumbnail_data_url="data:image/png;base64,xx",
            note="",
        )


class TestProviderProtocol:
    def test_recording_provider_receives_plan(self) -> None:
        provider = _RecordingProvider()
        gen = Scene3DGenerator(provider=provider)  # type: ignore[arg-type]
        plan, result = gen.generate(_analysis())
        assert provider.received == [plan]
        assert result.provider_status == "rendered"
        assert result.rendered_asset_url.endswith(".glb")
