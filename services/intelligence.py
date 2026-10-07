"""
Intelligence services for cross-case analysis.

This module extracts higher-level patterns (like 'Crime Series') from raw
evidence links using network analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
from networkx.algorithms.community import louvain_communities

from database.repository import list_cases, find_linked_cases


def get_crime_series(db_path: Path) -> dict[int, int]:
    """
    Identify 'Crime Series' by clustering cases based on shared evidence.

    Returns a mapping of case_id -> series_id.
    """
    cases = list_cases(db_path)
    if not cases:
        return {}

    # 1. Build the NetworkX Graph
    G = nx.Graph()

    # Add nodes
    for c in cases:
        G.add_node(c["case_id"])

    # Add weighted edges
    for c in cases:
        cid = c["case_id"]
        links = find_linked_cases(db_path, cid)
        for link in links:
            target_id = link["case_id"]
            # Only add edge once (since it's an undirected graph)
            if cid < target_id:
                G.add_edge(cid, target_id, weight=link["shared_count"])

    # 2. Automated Clustering (Louvain)
    community_map: dict[int, int] = {}
    if G.number_of_edges() > 0:
        communities = louvain_communities(G, weight="weight")
        for i, community in enumerate(communities):
            for node in community:
                community_map[node] = i
    else:
        # Each case is its own series if no links exist
        for node in G.nodes():
            community_map[node] = node

    return community_map
