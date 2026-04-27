"""Tool definitions for the network management agent."""

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from typing import Annotated

from .state import AgentState


def _deg2rad(df: pd.DataFrame) -> np.ndarray:
    """Convert lat/lon from degrees to radians."""
    return df[["lat", "lon"]].values * (np.pi / 180.0)


def _miles_to_radians(threshold_miles: float, earth_radius_miles: float = 3958.8) -> float:
    """Convert miles to radians for haversine distance."""
    return threshold_miles / earth_radius_miles


@tool
def get_candidates(
    specialty: str,
    candidates: Annotated[list[dict], InjectedState("candidates")],
    network: Annotated[list[dict], InjectedState("network")],
    sort_by: str = "none",
    ascending: bool = True,
):
    """Return up to 5 candidates with the given specialty that are not yet in the network.

    - sort_by: IMPORTANT - sorts the ENTIRE candidate pool before selecting the top 5.
      This means sort_by controls WHICH candidates you see, not just their display order.
      To prioritize certain providers, use sort_by with the column that indicates what
      the priority is (e.g. 'efficiency', 'quality').
    - ascending: if True, sort lowest-first; if False, sort highest-first.
    Returns a message if no candidates remain for that specialty.
    """
    network_df = pd.DataFrame(network) if network else pd.DataFrame()
    candidates_df = pd.DataFrame(candidates) if candidates else pd.DataFrame()

    used_ids = set(network_df.id) if not network_df.empty else set()

    filtered = candidates_df[
        (candidates_df.specialty == specialty) &
        (~candidates_df.id.isin(used_ids))
    ]

    if len(filtered) == 0:
        return "No available candidates for this specialty."

    if sort_by != "none":
        if sort_by in candidates_df.columns:
            filtered = filtered.sort_values(by=sort_by, ascending=ascending)
        else:
            return f"Invalid sort_by value '{sort_by}'. Valid columns are: {list(candidates_df.columns)}"

    if len(filtered) > 5:
        filtered = filtered.head(5)

    return filtered.to_dict(orient="records")


@tool
def get_candidate_schema(
    candidates: Annotated[list[dict], InjectedState("candidates")]
):
    """Return the schema of the candidate data as a dictionary mapping column names to their data types.

    Use this before calling get_candidates to identify valid sort_by column names,
    or to understand what filtering and prioritization options are available
    (e.g. ratings, distance, accepting_patients).
    """
    candidates_df = pd.DataFrame(candidates) if candidates else pd.DataFrame()

    if candidates_df.empty:
        return "No candidate data available."

    return candidates_df.dtypes.astype(str).to_dict()


@tool
def add_provider(
    id: int,
    network: Annotated[list[dict], InjectedState("network")],
    candidates: Annotated[list[dict], InjectedState("candidates")],
):
    """Add a provider to the network using the given provider ID.

    If the provider is already in the network, returns a message containing 'Skip'
    -- do not attempt to add them again.
    """
    network_df = pd.DataFrame(network) if network else pd.DataFrame()
    candidates_df = pd.DataFrame(candidates) if candidates else pd.DataFrame()

    if candidates_df.empty:
        return "No candidate data available."

    if not network_df.empty and id in network_df["id"].values:
        return f"Skip: Provider {id} is already in the network."

    match = candidates_df[candidates_df.id == id]
    if match.empty:
        return f"Provider {id} not found in candidates."

    new_row = match.iloc[0].to_dict()
    return new_row


@tool
def get_network_status(
    members: Annotated[list[dict], InjectedState("members")],
    network: Annotated[list[dict], InjectedState("network")],
    county_thresholds: Annotated[dict[str, float], InjectedState("county_thresholds")],
) -> dict[str, Any]:
    """Return the current network status including total providers and member coverage.

    - county_thresholds: A dictionary mapping county names to distance thresholds in miles.
      If a county is not listed, a default threshold of 20.0 miles is used.

    Output format:
      {
        "total_providers": int,
        "member_coverage": [
          {"county": str, "members_with_access": int, "total_members": int, "coverage_percentage": float},
          ...
        ]
      }
    """
    members_df = pd.DataFrame(members) if members else pd.DataFrame()
    net_df = pd.DataFrame(network) if network else pd.DataFrame()

    summary = {"total_providers": len(network)}
    if net_df.empty or members_df.empty:
        summary["member_coverage"] = []
        return summary

    required_cols = {"county", "lat", "lon"}
    if not required_cols.issubset(members_df.columns):
        raise KeyError(f"`members` must contain: {', '.join(sorted(required_cols))}")

    required_net_cols = {"lat", "lon"}
    if not required_net_cols.issubset(net_df.columns):
        raise KeyError(f"`network` must contain: {', '.join(sorted(required_net_cols))}")

    tree = BallTree(_deg2rad(net_df), leaf_size=40, metric="haversine")
    member_pts = _deg2rad(members_df)

    # We need to calculate coverage per county because thresholds can vary
    coverage_results = []
    
    # Group members by county
    for county, group in members_df.groupby("county"):
        threshold = county_thresholds.get(county, 20.0)
        radius_rad = _miles_to_radians(threshold)
        
        group_pts = _deg2rad(group)
        # Query radius for this specific group's points
        indices, _ = tree.query_radius(group_pts, r=radius_rad, return_distance=True)
        
        members_with_access = np.array([len(lst) > 0 for lst in indices]).sum()
        total_members = len(group)
        coverage_percentage = (members_with_access / total_members * 100).round(2)
        
        coverage_results.append({
            "county": county,
            "members_with_access": int(members_with_access),
            "total_members": int(total_members),
            "coverage_percentage": float(coverage_percentage)
        })

    # Sort by county name for consistency
    coverage_results.sort(key=lambda x: x["county"])
    summary["member_coverage"] = coverage_results
    return summary


TOOLS = [get_candidates, get_candidate_schema, add_provider, get_network_status]
