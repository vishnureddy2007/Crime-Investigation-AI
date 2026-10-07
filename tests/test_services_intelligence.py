"""
Tests for the intelligence service clustering logic.
"""

from __future__ import annotations

import pytest
from pathlib import Path
from database.db import init_db
from database.repository import save_case, save_analysis
from models.schemas import EvidenceAnalysis
from services.intelligence import get_crime_series

def _setup_case_with_labels(db_path: Path, source_name: str, labels: list[str]) -> int:
    """Helper to create a case with specific labels."""
    cid = save_case(db_path, source_name, "image")
    # Mock analysis with the given labels
    analysis = EvidenceAnalysis(
        source_name=source_name, source_type="image",
        counts_by_label={label: 1 for label in labels},
        total_objects=len(labels), unique_labels=labels,
        average_confidence=0.8, person_count=0, verified_weapon_count=0,
        candidate_weapon_count=0, weapon_count=0, vehicle_count=0, bag_count=0,
        severity_score=50, severity_level="high", suggested_category="generic",
        key_observations=[], has_threat=False, frame_count=1
    )
    save_analysis(db_path, cid, analysis)
    return cid

def test_get_crime_series_empty_db(tmp_path: Path) -> None:
    db_path = tmp_path / "test_empty.db"
    init_db(db_path)
    assert get_crime_series(db_path) == {}

def test_get_crime_series_single_case(tmp_path: Path) -> None:
    db_path = tmp_path / "test_single.db"
    init_db(db_path)
    cid = _setup_case_with_labels(db_path, "case1", ["knife"])
    res = get_crime_series(db_path)
    assert len(res) == 1
    assert cid in res

def test_get_crime_series_clear_clusters(tmp_path: Path) -> None:
    """Verify that cases sharing evidence are grouped together."""
    db_path = tmp_path / "test_clusters.db"
    init_db(db_path)

    # Cluster 1: Shared 'red_car'
    c1 = _setup_case_with_labels(db_path, "case1", ["red_car", "person"])
    c2 = _setup_case_with_labels(db_path, "case2", ["red_car", "bag"])

    # Cluster 2: Shared 'blue_bag'
    c3 = _setup_case_with_labels(db_path, "case3", ["blue_bag", "person"])
    c4 = _setup_case_with_labels(db_path, "case4", ["blue_bag", "knife"])

    res = get_crime_series(db_path)

    # Cases in the same cluster should have the same series_id
    assert res[c1] == res[c2]
    assert res[c3] == res[c4]
    # Different clusters should have different series_ids
    assert res[c1] != res[c3]

def test_get_crime_series_chain(tmp_path: Path) -> None:
    """Verify that a chain of evidence (A-B, B-C) forms one series."""
    db_path = tmp_path / "test_chain.db"
    init_db(db_path)

    # A shares with B, B shares with C. A does NOT share with C.
    c1 = _setup_case_with_labels(db_path, "case1", ["label_ab"])
    c2 = _setup_case_with_labels(db_path, "case2", ["label_ab", "label_bc"])
    c3 = _setup_case_with_labels(db_path, "case3", ["label_bc"])

    res = get_crime_series(db_path)

    # All should be in the same series
    assert res[c1] == res[c2] == res[c3]

def test_get_crime_series_isolated(tmp_path: Path) -> None:
    """Verify that cases with no shared evidence are isolated."""
    db_path = tmp_path / "test_isolated.db"
    init_db(db_path)

    c1 = _setup_case_with_labels(db_path, "case1", ["a"])
    c2 = _setup_case_with_labels(db_path, "case2", ["b"])
    c3 = _setup_case_with_labels(db_path, "case3", ["c"])

    res = get_crime_series(db_path)

    # Each should be in its own cluster
    ids = {res[c1], res[c2], res[c3]}
    assert len(ids) == 3
