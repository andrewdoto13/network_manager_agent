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


def _compute_coverage(
    network: list[dict],
    members: list[dict],
    county_thresholds: dict[str, float],
) -> list[dict]:
    """Compute per-county member coverage given a network and member set.

    Returns a list of dicts sorted by county name, each with keys:
        county, members_with_access, total_members, coverage_percentage
    """
    members_df = pd.DataFrame(members) if members else pd.DataFrame()
    net_df = pd.DataFrame(network) if network else pd.DataFrame()

    if net_df.empty or members_df.empty:
        return []

    required_cols = {"county", "lat", "lon"}
    if not required_cols.issubset(members_df.columns):
        raise KeyError(f"`members` must contain: {', '.join(sorted(required_cols))}")

    required_net_cols = {"lat", "lon"}
    if not required_net_cols.issubset(net_df.columns):
        raise KeyError(f"`network` must contain: {', '.join(sorted(required_net_cols))}")

    tree = BallTree(_deg2rad(net_df), leaf_size=40, metric="haversine")

    coverage_results = []
    for county, group in members_df.groupby("county"):
        threshold = county_thresholds.get(county, 20.0)
        radius_rad = _miles_to_radians(threshold)

        group_pts = _deg2rad(group)
        indices, _ = tree.query_radius(group_pts, r=radius_rad, return_distance=True)

        members_with_access = int(np.array([len(lst) > 0 for lst in indices]).sum())
        total_members = len(group)
        coverage_percentage = round(members_with_access / total_members * 100, 2)

        coverage_results.append({
            "county": county,
            "members_with_access": members_with_access,
            "total_members": total_members,
            "coverage_percentage": coverage_percentage,
        })

    coverage_results.sort(key=lambda x: x["county"])
    return coverage_results


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
    ids: list[int],
    network: Annotated[list[dict], InjectedState("network")],
    candidates: Annotated[list[dict], InjectedState("candidates")],
):
    """Add one or more providers to the network using their IDs.

    - ids: List of provider IDs to add. Pass a single ID as [id] or multiple as [id1, id2, ...].
    Returns a list of successfully added provider dicts. Providers already in the network
    are silently skipped. Invalid IDs are reported in the 'errors' field.
    """
    network_df = pd.DataFrame(network) if network else pd.DataFrame()
    candidates_df = pd.DataFrame(candidates) if candidates else pd.DataFrame()

    if candidates_df.empty:
        return "No candidate data available."

    used_ids = set(network_df["id"].values) if not network_df.empty else set()
    added: list[dict] = []
    errors: list[str] = []

    for pid in ids:
        if pid in used_ids:
            continue
        match = candidates_df[candidates_df.id == pid]
        if match.empty:
            errors.append(f"Provider {pid} not found in candidates.")
        else:
            added.append(match.iloc[0].to_dict())

    result: dict[str, Any] = {"added": added}
    if errors:
        result["errors"] = errors
    return result


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
    coverage = _compute_coverage(network, members, county_thresholds)
    return {
        "total_providers": len(network),
        "member_coverage": coverage,
    }


def _build_sim_network(
    network_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    add_ids: list[int],
    remove_ids: list[int],
) -> list[dict]:
    """Build a temporary network by applying additions and removals."""
    sim_network = [p for _, p in network_df.iterrows()] if not network_df.empty else []
    sim_network = [p.to_dict() if hasattr(p, "to_dict") else p for p in sim_network]

    for pid in remove_ids:
        sim_network = [p for p in sim_network if p["id"] != pid]

    for pid in add_ids:
        match = candidates_df[candidates_df.id == pid]
        if not match.empty:
            sim_network.append(match.iloc[0].to_dict())

    return sim_network


def _compute_delta(
    current_coverage: list[dict],
    simulated_coverage: list[dict],
) -> list[dict]:
    """Compute per-county coverage change between current and simulated."""
    current_map = {c["county"]: c["coverage_percentage"] for c in current_coverage}
    sim_map = {c["county"]: c["coverage_percentage"] for c in simulated_coverage}
    all_counties = sorted(set(list(current_map.keys()) + list(sim_map.keys())))
    delta = []
    for county in all_counties:
        change = round(sim_map.get(county, 0) - current_map.get(county, 0), 2)
        delta.append({
            "county": county,
            "coverage_change": change,
        })
    return delta


def _validate_scenario(
    add_ids: list[int],
    remove_ids: list[int],
    candidates_df: pd.DataFrame,
    network_df: pd.DataFrame,
) -> list[str]:
    """Validate a single scenario's provider IDs. Returns list of error strings."""
    errors: list[str] = []
    if len(add_ids) > 5:
        errors.append("add_ids: maximum 5 providers allowed per scenario.")
    if len(remove_ids) > 5:
        errors.append("remove_ids: maximum 5 providers allowed per scenario.")

    used_ids = set(network_df["id"].values) if not network_df.empty else set()

    for pid in add_ids:
        if pid in used_ids:
            errors.append(f"Provider {pid} is already in the network.")
        elif candidates_df.empty or pid not in candidates_df["id"].values:
            errors.append(f"Provider {pid} not found in candidates.")

    for pid in remove_ids:
        if network_df.empty or pid not in network_df["id"].values:
            errors.append(f"Provider {pid} is not in the current network.")

    return errors


@tool
def simulate_network_change(
    add_ids: list[int],
    remove_ids: list[int],
    candidates: Annotated[list[dict], InjectedState("candidates")],
    network: Annotated[list[dict], InjectedState("network")],
    members: Annotated[list[dict], InjectedState("members")],
    county_thresholds: Annotated[dict[str, float], InjectedState("county_thresholds")],
    compare_scenarios: list[dict] = [],
) -> dict[str, Any]:
    """Simulate adding or removing providers and show the coverage impact without modifying the network.

    Use this to test whether adding or removing providers would improve coverage before committing.

    MODE 1 - Single simulation:
        Pass add_ids and/or remove_ids to simulate one change.
        - add_ids: Provider IDs to add (max 5). Must exist in candidates and not be in the current network.
        - remove_ids: Provider IDs to remove (max 5). Must be in the current network.

    MODE 2 - Compare multiple scenarios:
        Pass compare_scenarios to test multiple mutually exclusive options in one call.
        Each scenario is evaluated independently and ranked by coverage improvement.
        - compare_scenarios: List of {"add_ids": [...], "remove_ids": [...]} dicts (max 5 scenarios).
        - When compare_scenarios is provided, add_ids and remove_ids are ignored.

    Returns current coverage, simulated coverage, and per-county delta (percentage point change).
    """
    candidates_df = pd.DataFrame(candidates) if candidates else pd.DataFrame()
    network_df = pd.DataFrame(network) if network else pd.DataFrame()

    current_coverage = _compute_coverage(network, members, county_thresholds)

    if compare_scenarios:
        if len(compare_scenarios) > 5:
            return {"error": "compare_scenarios: maximum 5 scenarios allowed."}

        all_errors: list[str] = []
        scenario_results: list[dict[str, Any]] = []

        for sc in compare_scenarios:
            sc_add = sc.get("add_ids", [])
            sc_remove = sc.get("remove_ids", [])
            errs = _validate_scenario(sc_add, sc_remove, candidates_df, network_df)
            if errs:
                all_errors.append({
                    "scenario": {"add_ids": sc_add, "remove_ids": sc_remove},
                    "errors": errs,
                })
                continue

            sim_net = _build_sim_network(network_df, candidates_df, sc_add, sc_remove)
            sim_cov = _compute_coverage(sim_net, members, county_thresholds)
            sc_delta = _compute_delta(current_coverage, sim_cov)

            scenario_results.append({
                "add_ids": sc_add,
                "remove_ids": sc_remove,
                "total_providers": len(sim_net),
                "coverage": sim_cov,
                "delta": sc_delta,
            })

        scenario_results.sort(
            key=lambda s: sum(d["coverage_change"] for d in s["delta"]),
            reverse=True,
        )
        for rank, sc in enumerate(scenario_results, 1):
            sc["rank"] = rank

        result: dict[str, Any] = {
            "current": {
                "total_providers": len(network),
                "coverage": current_coverage,
            },
            "scenarios": scenario_results,
        }
        if all_errors:
            result["errors"] = all_errors
        return result

    if len(add_ids) > 5:
        return {"error": "add_ids: maximum 5 providers allowed per simulation."}
    if len(remove_ids) > 5:
        return {"error": "remove_ids: maximum 5 providers allowed per simulation."}

    errors = _validate_scenario(add_ids, remove_ids, candidates_df, network_df)
    if errors:
        return {"error": "Invalid providers", "details": errors}

    sim_net = _build_sim_network(network_df, candidates_df, add_ids, remove_ids)
    sim_cov = _compute_coverage(sim_net, members, county_thresholds)
    delta = _compute_delta(current_coverage, sim_cov)

    return {
        "current": {
            "total_providers": len(network),
            "coverage": current_coverage,
        },
        "simulated": {
            "total_providers": len(sim_net),
            "coverage": sim_cov,
        },
        "delta": delta,
    }


TOOLS = [get_candidates, get_candidate_schema, add_provider, get_network_status, simulate_network_change]
