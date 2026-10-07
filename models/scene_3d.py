"""
3D crime-scene reconstruction pipeline (Phase 49).

Honest abstraction — NOT a fake 3D slideshow. The pipeline produces a
**structured plan** of the scene that an external 3D provider (e.g.
LumaAI / Meshy / Tripo3D) can consume to render an actual 3D asset.

Architecture
------------
1. `Scene3DObject`      — one entity in the scene (person, weapon, etc.)
2. `Scene3DCamera`      — camera viewpoint suggestion
3. `Scene3DPlan`        — full plan (objects + camera + scene bounds + meta)
4. `AI3DProvider`       — abstract interface an external 3D provider implements
5. `LocalProceduralProvider` — honest fallback that returns the plan
   unchanged and a status string "Local/demo mode — no 3D rendered".
   This is what runs when no external provider is configured.
6. `Scene3DGenerator`   — public facade that turns an `EvidenceAnalysis`
   into a `Scene3DPlan`, then asks the provider to render it.

Constraints
-----------
* Only VERIFIED weapons go into the plan (Phase 46 — a candidate weapon
  alone is never enough to invent a gun/knife in a 3D scene).
* No invented counts: if the analysis says 3 persons, the plan has 3
  persons — never 4 to "fill out the scene".
* The provider is the source of truth for "what was actually rendered".
  When the provider is local/demo, `provider_status == "local-fallback"`
  and `rendered_asset_url == ""`. The UI surfaces that honestly.
* The plan itself is serialisable to JSON — callers can persist it to
  the database for later auditing.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Protocol


# ----------------------------------------------------------------------
# Schemas
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Scene3DObject:
    """A single object in the planned 3D scene."""

    object_id: str             # stable id, e.g. "person_1", "weapon_2"
    kind: str                  # "person" | "weapon" | "vehicle" | "bag" | "generic"
    label: str                 # human-readable label
    verified: bool             # True iff the underlying detection was verified
    position_xyz: tuple[float, float, float]  # x, y, z in metres (origin at scene centre)
    size_xyz: tuple[float, float, float]      # bounding-box dimensions in metres
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Scene3DCamera:
    """Suggested camera setup for rendering the scene."""

    position_xyz: tuple[float, float, float]
    look_at_xyz: tuple[float, float, float]
    fov_degrees: float = 60.0


@dataclass
class Scene3DPlan:
    """Structured plan of the 3D scene — the contract between our
    pipeline and any external 3D provider."""

    plan_id: str
    source_name: str
    generated_at: datetime
    scene_bounds_xyz: tuple[float, float, float]
    objects: list[Scene3DObject]
    cameras: list[Scene3DCamera]
    warnings: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["generated_at"] = self.generated_at.isoformat(timespec="seconds")
        return d

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, default=str)

    @property
    def object_count(self) -> int:
        return len(self.objects)

    @property
    def verified_weapon_count(self) -> int:
        return sum(1 for o in self.objects if o.kind == "weapon" and o.verified)

    @property
    def candidate_weapon_count(self) -> int:
        return sum(1 for o in self.objects if o.kind == "weapon" and not o.verified)


# ----------------------------------------------------------------------
# Provider interface
# ----------------------------------------------------------------------
@dataclass
class Scene3DRenderResult:
    """What an external 3D provider returns."""

    provider_name: str
    provider_status: str       # "rendered" | "local-fallback" | "error"
    rendered_asset_url: str    # empty string in local-fallback mode
    thumbnail_data_url: str    # data URL of a low-res preview, "" if none
    note: str = ""             # human-readable explanation for the UI


class AI3DProvider(Protocol):
    """Protocol every 3D provider must satisfy.

    Real providers (LumaAI, Meshy, Tripo3D, etc.) would implement this
    and POST the plan to their API. The local fallback is `LocalProceduralProvider`.
    """

    name: str

    def render(self, plan: Scene3DPlan) -> Scene3DRenderResult: ...


class LocalProceduralProvider:
    """The honest fallback.

    Returns the plan *unchanged* with `provider_status="local-fallback"`.
    Does NOT pretend to render a 3D asset — that would be a fake 3D
    slideshow, which the project brief explicitly forbids.
    """

    name = "local-procedural"

    def __init__(self, *, note: str | None = None) -> None:
        self.note = note or (
            "Local/demo mode — no external 3D provider configured. "
            "The structured plan below is the real output; rendering a "
            "true 3D scene requires configuring AI3D_PROVIDER (see "
            "docs/USER_MANUAL.md)."
        )

    def render(self, plan: Scene3DPlan) -> Scene3DRenderResult:
        return Scene3DRenderResult(
            provider_name=self.name,
            provider_status="local-fallback",
            rendered_asset_url="",
            thumbnail_data_url="",
            note=self.note,
        )


# ----------------------------------------------------------------------
# Generator
# ----------------------------------------------------------------------
class Scene3DGenerator:
    """Build a `Scene3DPlan` from an `EvidenceAnalysis`.

    Parameters
    ----------
    provider:
        The 3D provider to render the plan. Defaults to
        `LocalProceduralProvider`. Pass a real provider implementation
        here when one is configured.
    """

    # Standard sizes in metres (rough but reasonable for a top-down scene).
    SIZES: dict[str, tuple[float, float, float]] = {
        "person":  (0.5, 1.7, 0.3),
        "weapon":  (0.3, 0.05, 0.05),
        "vehicle": (1.8, 1.5, 4.5),
        "bag":     (0.4, 0.3, 0.15),
    }

    def __init__(self, provider: AI3DProvider | None = None) -> None:
        self.provider: AI3DProvider = provider or LocalProceduralProvider()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_plan(self, analysis, *, base_image=None) -> Scene3DPlan:
        """
        Convert an EvidenceAnalysis into a structured 3D scene plan.

        Only VERIFIED weapons are included as weapon-kind objects;
        candidate weapons are surfaced in `warnings` for human review
        and are NOT placed into the scene.
        """
        warnings: list[str] = []

        # 1) Persons (one per verified person_count).
        persons = []
        for i in range(analysis.person_count):
            persons.append(
                Scene3DObject(
                    object_id=f"person_{i+1}",
                    kind="person",
                    label="Person",
                    verified=True,
                    position_xyz=self._spread(analysis.person_count, i, radius=1.5),
                    size_xyz=self.SIZES["person"],
                    metadata={"count_source": "person_count"},
                )
            )

        # 2) Vehicles.
        vehicles = []
        for i in range(analysis.vehicle_count):
            vehicles.append(
                Scene3DObject(
                    object_id=f"vehicle_{i+1}",
                    kind="vehicle",
                    label="Vehicle",
                    verified=True,
                    position_xyz=self._spread(analysis.vehicle_count, i, radius=3.0, y=0.0),
                    size_xyz=self.SIZES["vehicle"],
                    metadata={"count_source": "vehicle_count"},
                )
            )

        # 3) Bags.
        bags = []
        for i in range(analysis.bag_count):
            bags.append(
                Scene3DObject(
                    object_id=f"bag_{i+1}",
                    kind="bag",
                    label="Bag",
                    verified=True,
                    position_xyz=self._spread(analysis.bag_count, i, radius=2.0),
                    size_xyz=self.SIZES["bag"],
                    metadata={"count_source": "bag_count"},
                )
            )

        # 4) Weapons — VERIFIED ONLY. Candidates get a warning, never a mesh.
        weapons = []
        for i in range(analysis.verified_weapon_count):
            weapons.append(
                Scene3DObject(
                    object_id=f"weapon_{i+1}",
                    kind="weapon",
                    label="Weapon (verified)",
                    verified=True,
                    position_xyz=self._spread(analysis.verified_weapon_count, i, radius=1.0),
                    size_xyz=self.SIZES["weapon"],
                    metadata={"count_source": "verified_weapon_count"},
                )
            )
        if analysis.candidate_weapon_count > 0:
            warnings.append(
                f"{analysis.candidate_weapon_count} weapon candidate(s) were "
                "not placed in the 3D scene because they did not clear the "
                "verification threshold. Human review required."
            )

        # 5) Scene bounds = radius to fit everything plus a margin.
        max_radius = max(
            (1.5 if analysis.person_count else 0),
            (3.0 if analysis.vehicle_count else 0),
            (2.0 if analysis.bag_count else 0),
            (1.0 if analysis.verified_weapon_count else 0),
            2.0,  # always at least 2 m of scene
        )
        scene_bounds = (max_radius * 2 + 1.0, 2.5, max_radius * 2 + 1.0)

        # 6) Camera — pulled back and elevated, looking at origin.
        camera = Scene3DCamera(
            position_xyz=(0.0, max_radius + 2.0, max_radius + 4.0),
            look_at_xyz=(0.0, 0.0, 0.0),
            fov_degrees=60.0,
        )

        plan = Scene3DPlan(
            plan_id=self._plan_id(analysis),
            source_name=analysis.source_name,
            generated_at=datetime.now(),
            scene_bounds_xyz=scene_bounds,
            objects=persons + weapons + vehicles + bags,
            cameras=[camera],
            warnings=warnings,
            extra={
                "verified_weapon_count": analysis.verified_weapon_count,
                "candidate_weapon_count": analysis.candidate_weapon_count,
                "severity_level": analysis.severity_level,
                "suggested_category": analysis.suggested_category,
            },
        )
        return plan

    def generate(
        self,
        analysis,
        *,
        base_image=None,
    ) -> tuple[Scene3DPlan, Scene3DRenderResult]:
        """Build the plan and ask the provider to render it.

        Always returns BOTH the plan and the render result so the UI
        can show the plan even when the provider is in local/demo mode.
        """
        plan = self.build_plan(analysis, base_image=base_image)
        render_result = self.provider.render(plan)
        return plan, render_result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _spread(total: int, index: int, radius: float, y: float = 0.0) -> tuple[float, float, float]:
        """Spread `total` objects in a circle of `radius` metres."""
        if total <= 0:
            return (0.0, 0.0, 0.0)
        import math
        angle = (2 * math.pi * index) / max(1, total)
        return (round(radius * math.cos(angle), 2),
                y,
                round(radius * math.sin(angle), 2))

    @staticmethod
    def _plan_id(analysis) -> str:
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        safe = "".join(c if c.isalnum() or c in "-_" else "_"
                       for c in analysis.source_name.rsplit(".", 1)[0])[:32]
        return f"plan_{safe}_{ts}"
