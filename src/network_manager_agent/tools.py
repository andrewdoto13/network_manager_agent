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


def _aggregate_entities(candidates: list[dict]) -> pd.DataFrame:
    """Aggregate provider-level data into entity-level summaries.
    
    Returns a DataFrame indexed by 'Primary Contract Entity' with aggregated metrics.
    """
    if not candidates:
        return pd.DataFrame()
    
    df = pd.DataFrame(candidates)
    
    # Case-insensitive column discovery
    entity_col = next((c for c in df.columns if c.lower() == "primary contract entity"), "Primary Contract Entity")
    eff_col = next((c for c in df.columns if c.lower() == "effectiveness"), "Effectiveness")
    eta_col = next((c for c in df.columns if c.lower() == "efficiency"), "Efficiency")
    spec_col = next((c for c in df.columns if c.lower() == "specialty"), "Specialty")

    if entity_col not in df.columns:
        # If no entity column, treat each provider as its own entity
        df["Primary Contract Entity"] = df.index
        entity_col = "Primary Contract Entity"

    agg_map = {}
    if eff_col in df.columns:
        agg_map[eff_col] = "mean"
    if eta_col in df.columns:
        agg_map[eta_col] = "mean"
    if spec_col in df.columns:
        agg_map[spec_col] = lambda x: list(set(x.dropna()))
    
    # Always count providers per entity
    agg_map[entity_col] = "count"

    agg_df = df.groupby(entity_col).agg(agg_map)
    
    # Rename for consistency
    rename_map = {entity_col: "provider_count"}
    if eff_col in df.columns:
        rename_map[eff_col] = "avg_effectiveness"
    if eta_col in df.columns:
        rename_map[eta_col] = "avg_efficiency"
    if spec_col in df.columns:
        rename_map[spec_col] = "specialties"
        
    return agg_df.rename(columns=rename_map)


@tool
def get_candidates(
    specialty: str,
    candidates: Annotated[list[dict], InjectedState("candidates")],
    network: Annotated[list[dict], InjectedState("network")],
    sort_by: str = "none",
    ascending: bool = True,
):
    """Return up to 5 contract entities that have at least one provider with the given specialty and are not yet in the network.

    - sort_by: IMPORTANT - sorts the ENTITIES based on their aggregated metrics (e.g., 'avg_effectiveness', 'provider_count').
      To prioritize high-quality groups, use sort_by='avg_effectiveness' with ascending=False.
    - ascending: if True, sort lowest-first; if False, sort highest-first.
    Returns a message if no entities remain for that specialty.
    """
    candidates_df = pd.DataFrame(candidates) if candidates else pd.DataFrame()
    network_df = pd.DataFrame(network) if network else pd.DataFrame()

    if candidates_df.empty:
        return "No available candidates for this specialty."

    # Case-insensitive column discovery
    id_col = next((c for c in candidates_df.columns if c.lower() == "id"), "id")
    spec_col = next((c for c in candidates_df.columns if c.lower() == "specialty"), "specialty")
    entity_col = next((c for c in candidates_df.columns if c.lower() == "primary contract entity"), "Primary Contract Entity")

    used_ids = set(network_df[id_col]) if not network_df.empty and id_col in network_df.columns else set()
    
    # We consider an entity 'used' if ANY of its providers are in the network
    # But real-world payers usually contract the whole group. 
    # For this tool, let's assume an entity is in the network if it's already been added.
    # To be safe, we check if any provider of the entity is in the network.
    
    # 1. Find entities that have the specialty
    eligible_providers = candidates_df[candidates_df[spec_col] == specialty]
    if eligible_providers.empty:
        return "No available candidates for this specialty."
        
    eligible_entities = set(eligible_providers[entity_col].unique())
    
    # 2. Exclude entities that are already (partially) in the network
    if not network_df.empty and entity_col in network_df.columns:
        used_entities = set(network_df[entity_col].unique())
        eligible_entities = eligible_entities - used_entities

    if not eligible_entities:
        return "No available entities for this specialty."

    # 3. Get aggregated summaries for these entities
    entity_summaries = _aggregate_entities(candidates)
    filtered_entities = entity_summaries.loc[list(eligible_entities)]

    if sort_by != "none":
        if sort_by in filtered_entities.columns:
            filtered_entities = filtered_entities.sort_values(by=sort_by, ascending=ascending)
        else:
            return f"Invalid sort_by value '{sort_by}'. Valid entity metrics are: {list(filtered_entities.columns)}"

    if len(filtered_entities) > 5:
        filtered_entities = filtered_entities.head(5)

    # 4. Convert to Entity Summary Objects
    results = []
    for entity_id, row in filtered_entities.iterrows():
        results.append({
            "entity_id": entity_id,
            "metrics": {
                "avg_effectiveness": round(row["avg_effectiveness"], 2) if "avg_effectiveness" in row else None,
                "avg_efficiency": round(row["avg_efficiency"], 2) if "avg_efficiency" in row else None,
                "provider_count": int(row["provider_count"]),
            },
            "capabilities": {
                "specialties": row["specialties"] if "specialties" in row else [],
            },
            "summary": f"Entity '{entity_id}' has {int(row['provider_count'])} providers with an average effectiveness of {round(row['avg_effectiveness'], 2) if 'avg_effectiveness' in row else 'N/A'}."
        })

    return results



@tool
def get_candidate_schema(
    candidates: Annotated[list[dict], InjectedState("candidates")]
):
    """Return a rich statistical profile of the contract entity data.

    Use this before calling get_candidates to understand the available 
    entities, their average quality, and their size. This is essential 
    for deciding how to sort or filter entities.
    """
    entity_df = _aggregate_entities(candidates)

    if entity_df.empty:
        return "No candidate data available."

    profile = {}
    for col in entity_df.columns:
        dtype = str(entity_df[col].dtype)
        col_profile = {"type": dtype}

        if pd.api.types.is_numeric_dtype(entity_df[col]):
            col_profile.update({
                "min": float(entity_df[col].min()),
                "max": float(entity_df[col].max()),
                "mean": float(round(entity_df[col].mean(), 2)),
                "q1": float(entity_df[col].quantile(0.25)),
                "median": float(entity_df[col].median()),
                "q3": float(entity_df[col].quantile(0.75)),
            })
        elif pd.api.types.is_bool_dtype(entity_df[col]):
            counts = entity_df[col].value_counts().to_dict()
            col_profile["counts"] = {str(k): int(v) for k, v in counts.items()}
        elif pd.api.types.is_datetime64_any_dtype(entity_df[col]):
            col_profile.update({
                "min": str(entity_df[col].min()),
                "max": str(entity_df[col].max()),
                "samples": entity_df[col].head(3).dt.strftime('%Y-%m-%d').tolist(),
            })
        else:  # Categorical / Object
            if pd.api.types.is_list_like(entity_df[col].iloc[0]) if not entity_df[col].empty else False:
                # For list-like columns (e.g., specialties), we count total elements or unique elements across all lists
                all_vals = [item for sublist in entity_df[col].dropna() for item in sublist]
                unique_vals = set(all_vals)
                col_profile.update({
                    "unique_count": int(len(unique_vals)),
                    "distribution": {str(k): int(v) for k, v in pd.Series(all_vals).value_counts().head(5).to_dict().items()},
                })
            else:
                unique_values = entity_df[col].dropna().unique()
                col_profile.update({
                    "unique_count": int(len(unique_values)),
                    "distribution": {str(k): int(v) for k, v in entity_df[col].value_counts().head(5).to_dict().items()},
                })

        profile[col] = col_profile

    return profile


@tool
def add_contract_entity(
    entity_ids: list[str],
    network: Annotated[list[dict], InjectedState("network")],
    candidates: Annotated[list[dict], InjectedState("candidates")],
):
    """Add one or more contract entities to the network using their names.
    
    - entity_ids: List of entity names to add (e.g., ["Covenant Healthcare"]).
    Returns a list of all providers added to the network. Entities already in the network
    are silently skipped. Invalid entity names are reported in the 'errors' field.
    """
    network_df = pd.DataFrame(network) if network else pd.DataFrame()
    candidates_df = pd.DataFrame(candidates) if candidates else pd.DataFrame()

    if candidates_df.empty:
        return "No candidate data available."

    # Case-insensitive column discovery
    entity_col = next((c for c in candidates_df.columns if c.lower() == "primary contract entity"), "Primary Contract Entity")
    
    used_entities = set(network_df[entity_col].values) if not network_df.empty and entity_col in network_df.columns else set()
    added_providers: list[dict] = []
    errors: list[str] = []

    for eid in entity_ids:
        if eid in used_entities:
            continue
        
        match = candidates_df[candidates_df[entity_col] == eid] if entity_col in candidates_df.columns else pd.DataFrame()
        if match.empty:
            errors.append(f"Entity {eid} not found in candidates.")
        else:
            added_providers.extend(match.to_dict(orient="records"))

    result: dict[str, Any] = {"added_providers": added_providers}
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
    add_entity_ids: list[str],
    remove_entity_ids: list[str],
) -> list[dict]:
    """Build a temporary network by applying entity additions and removals."""
    sim_network = [p for _, p in network_df.iterrows()] if not network_df.empty else []
    sim_network = [p.to_dict() if hasattr(p, "to_dict") else p for p in sim_network]

    entity_col = next((c for c in candidates_df.columns if c.lower() == "primary contract entity"), "Primary Contract Entity")

    for eid in remove_entity_ids:
        sim_network = [p for p in sim_network if p.get(entity_col) != eid]

    for eid in add_entity_ids:
        match = candidates_df[candidates_df[entity_col] == eid] if entity_col in candidates_df.columns else pd.DataFrame()
        if not match.empty:
            sim_network.extend(match.to_dict(orient="records"))

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
    add_entity_ids: list[str],
    remove_entity_ids: list[str],
    candidates_df: pd.DataFrame,
    network_df: pd.DataFrame,
) -> list[str]:
    """Validate a single scenario's entity IDs. Returns list of error strings."""
    errors: list[str] = []
    if len(add_entity_ids) > 5:
        errors.append("add_entity_ids: maximum 5 entities allowed per scenario.")
    if len(remove_entity_ids) > 5:
        errors.append("remove_entity_ids: maximum 5 entities allowed per scenario.")

    entity_col = next((c for c in candidates_df.columns if c.lower() == "primary contract entity"), "Primary Contract Entity")
    used_entities = set(network_df[entity_col].values) if not network_df.empty and entity_col in network_df.columns else set()

    for eid in add_entity_ids:
        if eid in used_entities:
            errors.append(f"Entity {eid} is already in the network.")
        elif candidates_df.empty or (entity_col not in candidates_df.columns or eid not in candidates_df[entity_col].values):
            errors.append(f"Entity {eid} not found in candidates.")

    for eid in remove_entity_ids:
        if network_df.empty or (entity_col not in network_df.columns or eid not in network_df[entity_col].values):
            errors.append(f"Entity {eid} is not in the current network.")

    return errors



@tool
def simulate_network_change(
    add_entity_ids: list[str],
    remove_entity_ids: list[str],
    candidates: Annotated[list[dict], InjectedState("candidates")],
    network: Annotated[list[dict], InjectedState("network")],
    members: Annotated[list[dict], InjectedState("members")],
    county_thresholds: Annotated[dict[str, float], InjectedState("county_thresholds")],
    compare_scenarios: list[dict] = [],
) -> dict[str, Any]:
    """Simulate adding or removing entities and show the coverage impact without modifying the network.

    Use this to test whether adding or removing entities would improve coverage before committing.

    MODE 1 - Single simulation:
        Pass add_entity_ids and/or remove_entity_ids to simulate one change.
        - add_entity_ids: Entity names to add (max 5). Must exist in candidates and not be in the current network.
        - remove_entity_ids: Entity names to remove (max 5). Must be in the current network.

    MODE 2 - Compare multiple scenarios:
        Pass compare_scenarios to test multiple mutually exclusive options in one call.
        Each scenario is evaluated independently and ranked by coverage improvement.
        - compare_scenarios: List of {"add_entity_ids": [...], "remove_entity_ids": [...]} dicts (max 5 scenarios).
        - When compare_scenarios is provided, add_entity_ids and remove_entity_ids are ignored.

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
            sc_add = sc.get("add_entity_ids", [])
            sc_remove = sc.get("remove_entity_ids", [])
            errs = _validate_scenario(sc_add, sc_remove, candidates_df, network_df)
            if errs:
                all_errors.append({
                    "scenario": {"add_entity_ids": sc_add, "remove_entity_ids": sc_remove},
                    "errors": errs,
                })
                continue

            sim_net = _build_sim_network(network_df, candidates_df, sc_add, sc_remove)
            sim_cov = _compute_coverage(sim_net, members, county_thresholds)
            sc_delta = _compute_delta(current_coverage, sim_cov)

            scenario_results.append({
                "add_entity_ids": sc_add,
                "remove_entity_ids": sc_remove,
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

    if len(add_entity_ids) > 5:
        return {"error": "add_entity_ids: maximum 5 entities allowed per simulation."}
    if len(remove_entity_ids) > 5:
        return {"error": "remove_entity_ids: maximum 5 entities allowed per simulation."}

    errors = _validate_scenario(add_entity_ids, remove_entity_ids, candidates_df, network_df)
    if errors:
        return {"error": "Invalid entities", "details": errors}

    sim_net = _build_sim_network(network_df, candidates_df, add_entity_ids, remove_entity_ids)
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



TOOLS = [get_candidates, get_candidate_schema, add_contract_entity, get_network_status, simulate_network_change]
