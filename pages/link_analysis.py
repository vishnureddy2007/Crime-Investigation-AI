"""
Cross-Case Link Analysis — Crime Investigation AI.

Allows investigators to find patterns across multiple cases by identifying
shared evidence (labels) and calculating case similarities.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import uuid
from datetime import datetime

from config import DATABASE_PATH, APP_VERSION
from core.icons import ICON_DASHBOARD # Using dashboard icon as placeholder
from database.repository import (
    get_global_label_distribution,
    find_cases_by_label,
    find_linked_cases,
    list_cases,
    update_case_location,
    get_cases_with_locations,
    get_evidence_trends,
)
from models.schemas import IntelligenceReportData
from models.report_generator import IntelligenceReportGenerator
from pages._layout import render_page_header, empty_state
from services.intelligence import get_crime_series

def _render_global_stats() -> None:
    """Render global evidence distribution."""
    st.markdown("### Global Evidence Distribution")
    st.caption("Aggregated counts of all detected objects across the entire case repository.")

    dist = get_global_label_distribution(DATABASE_PATH)
    if not dist:
        st.info("No evidence detected across any cases yet.")
        return

    # Convert to DataFrame for better visualization
    df = pd.DataFrame([
        {"Label": k.title(), "Total Count": v}
        for k, v in dist.items()
    ]).sort_values("Total Count", ascending=False)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(df, use_container_width=True, hide_index=True)
    with col2:
        st.bar_chart(df, x="Label", y="Total Count")

def _render_label_search() -> None:
    """Search for all cases sharing a specific label."""
    st.markdown("---")
    st.markdown("### Label-Based Case Search")
    st.caption("Find every case in the database that contains a specific object.")

    # Get list of all labels for the dropdown
    dist = get_global_label_distribution(DATABASE_PATH)
    labels = sorted(list(dist.keys()))

    if not labels:
        st.info("No labels available to search.")
        return

    selected_label = st.selectbox("Select Object Label", options=labels, format_func=lambda x: x.title())

    if selected_label:
        matches = find_cases_by_label(DATABASE_PATH, selected_label)
        if matches:
            st.success(f"Found {len(matches)} case(s) containing '{selected_label}'.")
            df_matches = pd.DataFrame(matches)
            st.table(df_matches)
        else:
            st.info("No cases found with this label.")

def _render_visual_graph(ignored_labels: list[str] | None = None) -> None:
    """Render an interactive network graph of cases and their shared evidence."""
    st.markdown("---")
    st.markdown("### Visual Crime Series Network")
    st.caption("Interactive map of cases. Colors represent 'Crime Series' (automated clusters). Line thickness represents shared evidence strength.")

    cases = list_cases(DATABASE_PATH)
    if not cases:
        return

    # 1. Build the NetworkX Graph
    G = nx.Graph()

    # Add nodes
    for c in cases:
        G.add_node(c["case_id"], label=f"#{c['case_id']}: {c['source_name']}")

    # Add weighted edges
    for c in cases:
        cid = c["case_id"]
        links = find_linked_cases(DATABASE_PATH, cid, exclude_labels=ignored_labels)
        for link in links:
            target_id = link["case_id"]
            # Only add edge once (since it's an undirected graph)
            if cid < target_id:
                G.add_edge(cid, target_id, weight=link["shared_count"])

    if G.number_of_edges() == 0:
        st.info("No shared evidence found between cases. The network consists of isolated incidents.")
        # We still render the isolated nodes
        pass

    # 2. Automated Clustering (Crime Series Detection)
    community_map = get_crime_series(DATABASE_PATH)

    # --- Export Logic ---
    col_graph, col_btn = st.columns([4, 1])
    with col_btn:
        if st.button("📥 Export Intelligence Report", use_container_width=True):
            # Prepare data for report
            report_data = IntelligenceReportData(
                report_id=f"INTEL-{uuid.uuid4().hex[:8].upper()}",
                generated_at=datetime.now(),
                global_distribution=get_global_label_distribution(DATABASE_PATH),
                crime_series=get_crime_series(DATABASE_PATH),
                case_details={c["case_id"]: c["source_name"] for c in cases},
                app_version=APP_VERSION
            )

            pdf_bytes = IntelligenceReportGenerator().generate(report_data)
            st.download_button(
                label="Download PDF",
                data=pdf_bytes,
                file_name=f"Intel_Report_{report_data.report_id}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

    with col_graph:
        # 3. Visualization with Plotly
        pos = nx.spring_layout(G, k=0.5, seed=42)

        # Edge Traces
        edge_x = []
        edge_y = []
        edge_widths = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            # Scale width for visibility
            edge_widths.append(edge[2].get("weight", 1) * 2)

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color="#888"),
            hoverinfo='none',
            mode='lines',
        )

        # Node Traces
        node_x = []
        node_y = []
        node_colors = []
        node_texts = []
        node_sizes = []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            series_id = community_map.get(node, 0)
            node_colors.append(series_id)

            label = G.nodes[node].get("label", str(node))
            node_texts.append(label)

            # Size based on degree (how many connections it has)
            deg = G.degree(node)
            node_sizes.append(15 + (deg * 5))

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_texts,
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                reversescale=True,
                color=node_colors,
                size=node_sizes,
                colorbar=dict(
                    thickness=15,
                    title='Crime Series ID',
                    xanchor='left',
                ),
                line_width=2
            )
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=0),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                    ))

        st.plotly_chart(fig, use_container_width=True)

def _render_similarity_analysis(ignored_labels: list[str] | None = None) -> None:
    """Render a similarity matrix/table of cases based on shared evidence."""
    st.markdown("---")
    st.markdown("### 📊 Case Similarity Analysis")
    st.caption("Deep-dive into the relationship between specific cases. Based on overlapping evidence labels.")

    cases = list_cases(DATABASE_PATH)
    if not cases:
        return

    case_labels = [f"#{c['case_id']} — {c['source_name']}" for c in cases]
    selected_idx = st.selectbox("Select a Case to Analyze Similarity", options=range(len(cases)), format_func=lambda i: case_labels[i])

    if selected_idx is not None:
        target_case = cases[selected_idx]
        cid = target_case["case_id"]

        links = find_linked_cases(DATABASE_PATH, cid, exclude_labels=ignored_labels)

        if links:
            # Convert links to a readable DataFrame
            similarity_rows = []
            for link in links:
                # Find source name for the linked case
                linked_case_name = "Unknown"
                for c in cases:
                    if c["case_id"] == link["case_id"]:
                        linked_case_name = c["source_name"]
                        break

                similarity_rows.append({
                    "Case ID": f"#{link['case_id']}",
                    "Source Name": linked_case_name,
                    "Shared Evidence Count": link["shared_count"],
                })

            df_sim = pd.DataFrame(similarity_rows)
            st.table(df_sim)
        else:
            st.info("No similar cases found based on the current filters.")


def _render_temporal_analysis() -> None:
    """Render a time-series trend of detections."""
    st.markdown("---")
    st.markdown("### 📈 Temporal Evidence Trends")
    st.caption("Track how different object detections evolve over time across all cases.")

    trends = get_evidence_trends(DATABASE_PATH)
    if not trends:
        st.info("No temporal data available.")
        return

    # Prepare data for plotting
    dates = [t["date"] for t in trends]
    labels = set()
    for t in trends:
        labels.update(t["labels"].keys())

    # Create a DataFrame for Plotly
    plot_data = []
    for t in trends:
        for label in labels:
            plot_data.append({
                "Date": t["date"],
                "Label": label.title(),
                "Count": t["labels"].get(label, 0)
            })
    df_trends = pd.DataFrame(plot_data)

    import plotly.express as px
    fig = px.line(
        df_trends, x="Date", y="Count", color="Label",
        title="Evidence Frequency Over Time",
        markers=True,
        template="plotly_white"
    )
    fig.update_layout(xaxis_title="Date", yaxis_title="Detections")
    st.plotly_chart(fig, use_container_width=True)

def _render_spatial_intelligence() -> None:
    """Render a map of case locations and a tool to set them."""
    st.markdown("---")
    st.markdown("### 🗺️ Spatial Intelligence")
    st.caption("Geographic distribution of cases. Use the tool below to assign locations to cases.")

    col_map, col_tool = st.columns([3, 1])

    with col_tool:
        st.markdown("**Assign Case Location**")
        cases = list_cases(DATABASE_PATH)
        if cases:
            case_labels = [f"#{c['case_id']} — {c['source_name']}" for c in cases]
            sel_case = st.selectbox("Select Case", options=range(len(cases)), format_func=lambda i: case_labels[i])
            lat = st.number_input("Latitude", format="%.6f", value=0.0)
            lon = st.number_input("Longitude", format="%.6f", value=0.0)
            if st.button("Save Location", use_container_width=True):
                update_case_location(DATABASE_PATH, cases[sel_case]["case_id"], lat, lon)
                st.success("Location updated!")
                st.rerun()
        else:
            st.info("No cases available.")

    with col_map:
        located_cases = get_cases_with_locations(DATABASE_PATH)
        if located_cases:
            map_df = pd.DataFrame(located_cases)
            st.map(map_df[['latitude', 'longitude']])
        else:
            st.info("No case locations assigned. Use the tool on the right to add some.")

def render() -> None:
    """Render the Link Analysis page."""
    render_page_header(
        "🔗 Link Analysis",
        subtitle="Cross-Case Intelligence — Identify patterns, shared evidence, and similarities across the entire case repository.",
    )

    # Ensure we have some cases
    if not list_cases(DATABASE_PATH):
        empty_state(
            "No cases available for analysis.",
            "Process some evidence in the Investigation workspace to populate the repository.",
            action_label="Open Investigation Workspace",
            action_target="pages/investigation.py",
        )
        return

    # --- Intelligence Filters ---
    st.markdown("### 🔍 Intelligence Filters")
    dist = get_global_label_distribution(DATABASE_PATH)
    labels = sorted(list(dist.keys()))

    col1, col2 = st.columns([2, 1])
    with col1:
        ignored_labels = st.multiselect(
            "Ignore Common Labels (Noise Reduction)",
            options=labels,
            format_func=lambda x: x.title(),
            help="Exclude common objects (e.g., 'person') to find higher-quality links between cases."
        )
    with col2:
        st.caption("Filtering helps isolate rare evidence that strongly links cases.")

    _render_global_stats()
    _render_label_search()
    _render_visual_graph(ignored_labels)
    _render_similarity_analysis(ignored_labels)
    _render_temporal_analysis()
    _render_spatial_intelligence()
