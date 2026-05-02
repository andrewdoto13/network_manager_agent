"""Tool definitions for the network management agent."""

import json
import math
import signal
import functools
import itertools
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from typing import Annotated

from .data import normalize_coordinates, normalize_records


# ---------------------------------------------------------------------------
# Low-level geographic utilities
# ---------------------------------------------------------------------------

def _deg2rad(df: pd.DataFrame) -> np.ndarray:
    """Convert lat/lon from degrees to radians."""
    return df[["lat", "lon"]].values * (np.pi / 180.0)


def _miles_to_radians(threshold_miles: float, earth_radius_miles: float = 3958.8) -> float:
    """Convert miles to radians for haversine distance."""
    return threshold_miles / earth_radius_miles


# ---------------------------------------------------------------------------
# Coverage computation
# ---------------------------------------------------------------------------

def compute_coverage_df(
    network_df: pd.DataFrame,
    members_df: pd.DataFrame,
    county_specialty_thresholds: dict[str, dict[str, dict[str, float]]],
    candidates_df: pd.DataFrame = None,
) -> tuple[list[dict], list[str]]:
    """Compute per-county-and-specialty member coverage given DataFrames.
    
    Thresholds use nested structure: {"mi": {"wayne": {"general practice": 20.0}}}.
    """
    if members_df.empty:
        return [], []

    if "county" not in members_df.columns:
        return [], ["Required column 'county' not found in members data."]

    # Discover specialty column in candidates
    spec_col = None
    valid_specialties_lower = set()
    if candidates_df is not None and not candidates_df.empty:
        spec_col = "specialty" if "specialty" in candidates_df.columns else None
        if spec_col:
            valid_specialties_lower = set(candidates_df[spec_col].dropna().str.lower().unique().tolist())

    # Validate specialties against candidate data
    validation_errors = []
    for state_val, counties in county_specialty_thresholds.items():
        for county_val, specialties in counties.items():
            for spec in specialties.keys():
                if valid_specialties_lower and spec.lower() not in valid_specialties_lower:
                    validation_errors.append(
                        f"Specialty '{spec}' in {state_val}/{county_val} not found in candidate data."
                    )

    coverage_results = []
    for state_val, counties in county_specialty_thresholds.items():
        for county_val, specialties in counties.items():
            group_pts = _deg2rad(members_df)

            for specialty, threshold in specialties.items():
                # Filter network providers by specialty (case-insensitive)
                if spec_col and "specialty" in network_df.columns:
                    specialty_network = network_df[network_df[spec_col].str.lower() == specialty.lower()]
                else:
                    specialty_network = pd.DataFrame()

                if specialty_network.empty:
                    coverage_results.append({
                        "state": state_val,
                        "county": county_val,
                        "specialty": specialty,
                        "members_with_access": 0,
                        "total_members": len(members_df),
                        "coverage_percentage": 0.0,
                    })
                    continue

                radius_rad = _miles_to_radians(threshold)
                tree = BallTree(_deg2rad(specialty_network), leaf_size=40, metric="haversine")
                indices, _ = tree.query_radius(group_pts, r=radius_rad, return_distance=True)

                members_with_access = int(np.array([len(lst) > 0 for lst in indices]).sum())
                total_members = len(members_df)
                coverage_percentage = round(members_with_access / total_members * 100, 2)

                coverage_results.append({
                    "state": state_val,
                    "county": county_val,
                    "specialty": specialty,
                    "members_with_access": members_with_access,
                    "total_members": total_members,
                    "coverage_percentage": coverage_percentage,
                })

    coverage_results.sort(key=lambda x: (x.get("state", ""), x["county"], x["specialty"]))
    return coverage_results, validation_errors


def _compute_coverage(
    network: list[dict],
    members: list[dict],
    county_specialty_thresholds: dict[str, dict[str, dict[str, float]]],
    candidates: list[dict] = None,
) -> tuple[list[dict], list[str]]:
    """Compute per-county-and-specialty member coverage given a network and member set.
    
    Members are expected to already be scoped to the service area.
    Thresholds use nested structure: {"mi": {"wayne": {"general practice": 20.0}}}.
    """
    try:
        members_df = pd.DataFrame(members) if members else pd.DataFrame()
        net_df = pd.DataFrame(network) if network else pd.DataFrame()
        candidates_df = pd.DataFrame(candidates) if candidates else pd.DataFrame()
    except Exception as e:
        return [], [f"Data preparation error: {str(e)}"]

    return compute_coverage_df(net_df, members_df, county_specialty_thresholds, candidates_df)


def _filter_by_service_area(
    candidates: list[dict],
    members: list[dict],
    county_specialty_thresholds: dict[str, dict[str, dict[str, float]]],
) -> tuple[list[dict], list[dict]]:
    """Filter candidates and members to the defined service area.

    Thresholds use nested structure: {"mi": {"wayne": {"general practice": 20.0}}}.
    Members are filtered by state + county. Candidates are filtered to entities
    with at least one provider within threshold+buffer distance of the scoped members.

    Returns:
        Tuple of (filtered_candidates, filtered_members).
    """
    candidates = normalize_records(candidates) if candidates else []
    members = normalize_records(members) if members else []
    if not candidates or not members:
        return candidates, members

    if not county_specialty_thresholds:
        return candidates, members

    # Extract all states and counties from thresholds
    state_counties: dict[str, set[str]] = {}
    all_thresholds: dict[str, dict[str, float]] = {}
    for state_val, counties in county_specialty_thresholds.items():
        state_lower = state_val.lower()
        state_counties[state_lower] = set()
        for county_val, specs in counties.items():
            county_lower = county_val.lower()
            state_counties[state_lower].add(county_lower)
            all_thresholds[f"{state_lower}:{county_lower}"] = specs

    max_threshold = max(
        thresh
        for specs in all_thresholds.values()
        for thresh in specs.values()
    )
    buffer_deg = (max_threshold + 20) / 69.0

    # Filter members by state + county
    members_df = pd.DataFrame(members)
    if "state" in members_df.columns:
        members_df["state_lower"] = members_df["state"].astype(str).str.lower()
        members_df["county_lower"] = members_df["county"].astype(str).str.lower()
        mask = members_df["state_lower"].isin(state_counties.keys())
        for state_lower, counties in state_counties.items():
            mask = mask | (
                (members_df["state_lower"] == state_lower)
                & (members_df["county_lower"].isin(counties))
            )
        members_df = members_df[mask].drop(columns=["state_lower", "county_lower"], errors="ignore")
    else:
        # No state column — filter by county only (backward compat)
        all_counties = set()
        for counties in state_counties.values():
            all_counties.update(counties)
        members_df = members_df[members_df["county"].str.lower().isin(all_counties)]

    if members_df.empty:
        return [], members

    # Compute bounding box from scoped members
    lat_min = members_df["lat"].min() - buffer_deg
    lat_max = members_df["lat"].max() + buffer_deg
    lon_min = members_df["lon"].min() - buffer_deg
    lon_max = members_df["lon"].max() + buffer_deg

    # Filter candidates
    cdf = pd.DataFrame(candidates)
    normalize_coordinates(cdf)

    in_bounds_mask = (
        (cdf["lat"] >= lat_min)
        & (cdf["lat"] <= lat_max)
        & (cdf["lon"] >= lon_min)
        & (cdf["lon"] <= lon_max)
    )

    qualifying_entities = set(
        cdf.loc[in_bounds_mask, "entity"].dropna().unique().tolist()
    )

    entity_mask = cdf["entity"].isin(qualifying_entities)
    filtered_candidates = cdf.loc[entity_mask].reset_index(drop=True).to_dict(orient="records")
    filtered_members = members_df.to_dict(orient="records")

    return filtered_candidates, filtered_members


# ---------------------------------------------------------------------------
# Entity aggregation
# ---------------------------------------------------------------------------

def _aggregate_entities(candidates: list[dict]) -> pd.DataFrame:
    """Aggregate provider-level data into entity-level summaries.
    
    Returns a DataFrame indexed by 'entity' with aggregated metrics
    including effectiveness, efficiency, specialties, provider count, claims volume,
    and location confidence distribution.
    """
    if not candidates:
        return pd.DataFrame()
    
    df = pd.DataFrame(candidates)
    
    # Use canonical column names (already normalized at load time)
    entity_col = "entity" if "entity" in df.columns else None
    eff_col = "effectiveness" if "effectiveness" in df.columns else None
    eta_col = "efficiency" if "efficiency" in df.columns else None
    spec_col = "specialty" if "specialty" in df.columns else None
    claims_col = "total_claims_amount" if "total_claims_amount" in df.columns else None
    medicare_claims_col = "medicare_total_claims_amount" if "medicare_total_claims_amount" in df.columns else None
    confidence_col = "location_confidence" if "location_confidence" in df.columns else None
    new_pat_col = "new_patient_claims" if "new_patient_claims" in df.columns else None
    medicare_claims_vol_col = "medicare_claims_volume" if "medicare_claims_volume" in df.columns else None
    total_claims_vol_col = "total_claims_volume" if "total_claims_volume" in df.columns else None
    city_col = "city" if "city" in df.columns else None

    if entity_col is None:
        df["entity"] = df.index
        entity_col = "entity"

    agg_map = {}
    if eff_col:
        agg_map[eff_col] = "mean"
    if eta_col:
        agg_map[eta_col] = "mean"
    if spec_col:
        agg_map[spec_col] = lambda x: list(set(x.dropna()))
    
    if new_pat_col:
        agg_map[new_pat_col] = lambda x: float(round((x == 'Yes').mean() * 100, 2)) if not x.empty else None
    if medicare_claims_vol_col:
        def vol_dist(x):
            dist = x.dropna().value_counts().to_dict()
            return {k: int(v) for k, v in dist.items()}
        agg_map[medicare_claims_vol_col] = vol_dist
    if total_claims_vol_col:
        def vol_dist(x):
            dist = x.dropna().value_counts().to_dict()
            return {k: int(v) for k, v in dist.items()}
        agg_map[total_claims_vol_col] = vol_dist
    if city_col:
        agg_map[city_col] = "nunique"

    # Always count providers per entity
    agg_map[entity_col] = "count"

    # Numeric aggregation for claims volume
    if claims_col:
        agg_map[claims_col] = lambda x: float(round(x.dropna().mean(), 2)) if x.dropna().any() else None
    if medicare_claims_col:
        agg_map[medicare_claims_col] = lambda x: float(round(x.dropna().mean(), 2)) if x.dropna().any() else None

    # Categorical distributions for location confidence
    if confidence_col:
        def confidence_dist(x):
            dist = x.dropna().value_counts().to_dict()
            return {k: int(v) for k, v in dist.items()}
        agg_map[confidence_col] = confidence_dist

    agg_df = df.groupby(entity_col).agg(agg_map)

    # Compute total (sum) claims per entity separately and merge
    if claims_col:
        totals = df.groupby(entity_col)[claims_col].apply(
            lambda x: float(round(x.dropna().sum(), 2)) if x.dropna().any() else None
        )
        totals.name = "_sum_total_claims_amount"
        agg_df = agg_df.join(totals)
    if medicare_claims_col:
        totals = df.groupby(entity_col)[medicare_claims_col].apply(
            lambda x: float(round(x.dropna().sum(), 2)) if x.dropna().any() else None
        )
        totals.name = "_sum_medicare_total_claims_amount"
        agg_df = agg_df.join(totals)

    return agg_df.rename(columns={
        entity_col: "provider_count",
        eff_col: "avg_effectiveness" if eff_col else None,
        eta_col: "avg_efficiency" if eta_col else None,
        spec_col: "specialties" if spec_col else None,
        claims_col: "avg_total_claims_amount" if claims_col else None,
        medicare_claims_col: "avg_medicare_total_claims_amount" if medicare_claims_col else None,
        confidence_col: "location_confidence_dist" if confidence_col else None,
        new_pat_col: "new_patient_rate" if new_pat_col else None,
        medicare_claims_vol_col: "medicare_claims_volume_dist" if medicare_claims_vol_col else None,
        total_claims_vol_col: "total_claims_volume_dist" if total_claims_vol_col else None,
        city_col: "geographic_reach" if city_col else None,
        "_sum_total_claims_amount": "total_claims_amount",
        "_sum_medicare_total_claims_amount": "total_medicare_claims_amount",
    })


def _build_schema_profile_from_summaries(entity_df: pd.DataFrame) -> dict:
    """Build a statistical profile from a pre-aggregated entity DataFrame."""
    profile = {}
    
    # Exclude the entity identifier column from the profile
    id_cols = {"entity", "entity_id"}
    cols_to_profile = [col for col in entity_df.columns if col not in id_cols]
    
    for col in cols_to_profile:
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
                "samples": entity_df[col].head(3).dt.strftime("%Y-%m-%d").tolist(),
            })
        else:
            first_val = entity_df[col].dropna().iloc[0] if not entity_df[col].dropna().empty else None
            if isinstance(first_val, dict):
                full_dist = {}
                for val in entity_df[col].dropna():
                    if isinstance(val, dict):
                        for k, v in val.items():
                            full_dist[str(k)] = full_dist.get(str(k), 0) + v
                col_profile.update({
                    "type": "dict",
                    "distribution": {k: int(v) for k, v in full_dist.items()},
                    "num_keys": int(len(full_dist)),
                })
            elif pd.api.types.is_list_like(first_val):
                all_vals = [item for sublist in entity_df[col].dropna() for item in sublist]
                unique_vals = sorted(list(set(all_vals)))
                col_profile.update({
                    "unique_count": int(len(unique_vals)),
                    "unique_values": unique_vals,
                    "distribution": {
                        str(k): int(v)
                        for k, v in pd.Series(all_vals).value_counts().head(5).to_dict().items()
                    },
                })
            else:
                unique_values = sorted(entity_df[col].dropna().unique().tolist())
                col_profile.update({
                    "unique_count": int(len(unique_values)),
                    "unique_values": unique_values,
                    "distribution": {
                        str(k): int(v)
                        for k, v in entity_df[col].value_counts().head(5).to_dict().items()
                    },
                })

        profile[col] = col_profile

    return profile


# ---------------------------------------------------------------------------
# Public precompute helpers
# ---------------------------------------------------------------------------

def precompute_entity_summaries(candidates: list[dict]) -> list[dict]:
    """Pre-compute aggregated entity summaries from candidate data.

    Returns a list of dicts (one per entity) suitable for direct injection
    into AgentState and tool lookups.
    """
    if not candidates:
        return []
    candidates = normalize_records(candidates)
    agg_df = _aggregate_entities(candidates)
    if agg_df.empty:
        return []
    return agg_df.reset_index().to_dict(orient="records")


def get_candidate_schema_profile(
    candidates: list[dict] = None,
    entity_summaries: list[dict] = None,
) -> dict:
    """Compute a rich statistical profile of the contract entity data.

    Accepts either pre-computed entity_summaries or raw candidates.
    If entity_summaries are provided, they are used directly.
    Otherwise, candidates are aggregated first.
    """
    if entity_summaries:
        entity_df = pd.DataFrame(entity_summaries)
    elif candidates:
        entity_df = _aggregate_entities(candidates)
    else:
        return {}

    if entity_df.empty:
        return {}

    return _build_schema_profile_from_summaries(entity_df)


def get_raw_candidate_schema_profile(candidates: list[dict]) -> dict:
    """Build a statistical profile of raw provider-level candidate data.

    Same format as _build_schema_profile_from_summaries but for provider-level columns.
    Returns dict with column names as keys and statistical profiles as values.
    Does not exclude the 'entity' column since raw data needs it for grouping.
    Limits unique_values to first 20 to keep schema size manageable.
    """
    if not candidates:
        return {}
    df = pd.DataFrame(candidates)
    profile = {}
    
    # Don't exclude 'entity' for raw data - agent needs it for groupby
    cols_to_profile = [col for col in df.columns if col not in {"entity_id"}]
    
    for col in cols_to_profile:
        dtype = str(df[col].dtype)
        col_profile = {"type": dtype}

        if pd.api.types.is_numeric_dtype(df[col]):
            col_profile.update({
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(round(df[col].mean(), 2)),
                "q1": float(df[col].quantile(0.25)),
                "median": float(df[col].median()),
                "q3": float(df[col].quantile(0.75)),
            })
        elif pd.api.types.is_bool_dtype(df[col]):
            counts = df[col].value_counts().to_dict()
            col_profile["counts"] = {str(k): int(v) for k, v in counts.items()}
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_profile.update({
                "min": str(df[col].min()),
                "max": str(df[col].max()),
                "samples": df[col].head(3).dt.strftime("%Y-%m-%d").tolist(),
            })
        else:
            first_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            if isinstance(first_val, dict):
                full_dist = {}
                for val in df[col].dropna():
                    if isinstance(val, dict):
                        for k, v in val.items():
                            full_dist[str(k)] = full_dist.get(str(k), 0) + v
                col_profile.update({
                    "type": "dict",
                    "distribution": {k: int(v) for k, v in full_dist.items()},
                    "num_keys": int(len(full_dist)),
                })
            elif pd.api.types.is_list_like(first_val):
                all_vals = [item for sublist in df[col].dropna() for item in sublist]
                unique_vals = sorted(list(set(all_vals)))
                col_profile.update({
                    "unique_count": int(len(unique_vals)),
                    "unique_values": unique_vals[:20],  # Cap to first 20
                    "distribution": {
                        str(k): int(v)
                        for k, v in pd.Series(all_vals).value_counts().head(5).to_dict().items()
                    },
                })
            else:
                unique_values = sorted(df[col].dropna().unique().tolist())
                col_profile.update({
                    "unique_count": int(len(unique_values)),
                    "unique_values": unique_values[:20],  # Cap to first 20
                    "distribution": {
                        str(k): int(v)
                        for k, v in df[col].value_counts().head(5).to_dict().items()
                    },
                })

        profile[col] = col_profile

    return profile


# ---------------------------------------------------------------------------
# LangGraph Tools
# ---------------------------------------------------------------------------

@tool
def get_candidates(
    candidates: Annotated[list[dict], InjectedState("candidates")],
    network: Annotated[list[dict], InjectedState("network")],
    entity_summaries: Annotated[list[dict], InjectedState("entity_summaries")],
    specialties: list[str] = None,
    sort_by: str = "none",
    ascending: bool = True,
    limit: int = 5,
    offset: int = 0,
    weighted_metrics: dict[str, float] = None,
    search: str = None,
    exclude_entity_ids: list[str] = None,
    min_effectiveness: float = None,
    min_provider_count: int = None,
    max_geographic_reach: int = None,
    include_not_in_network: bool = True,
):
    """Return contract entities matching the specified filters, not yet in the network.

    This tool supports rich filtering and pagination to help the agent discover
    the best entities for the network.

    FILTERS:
    - specialties: List of specialty names to filter by (case-insensitive, exact match).
      Pass multiple specialties to get entities matching ANY of them.
    - search: Free-text search that matches against entity names, specialties, and cities.
      Use this to find entities by name fragment or to discover what's available.
    - exclude_entity_ids: List of entity names to exclude from results (e.g., already considered).
    - min_effectiveness: Minimum avg_effectiveness threshold (0-5 scale).
    - min_provider_count: Minimum number of providers in the entity.
    - max_geographic_reach: Maximum number of distinct cities the entity operates in.
    - include_not_in_network: If True (default), exclude entities already in the network.
      Set to False to see all entities including ones already contracted.

    SORTING:
    - sort_by: Sort entities by an aggregated metric. Options include:
      'avg_effectiveness', 'avg_efficiency', 'provider_count', 'new_patient_rate',
      'total_claims_amount', 'avg_total_claims_amount', 'total_medicare_claims_amount',
      'geographic_reach'. Use 'none' to return in natural order.
    - ascending: If True, sort lowest-first; if False, sort highest-first.
    - weighted_metrics: A dictionary of {metric_name: weight} for a custom score.
      Example: {"avg_effectiveness": 0.7, "new_patient_rate": 0.3}.

    PAGINATION:
    - limit: Maximum number of entities to return (default 5, max 100).
    - offset: Skip this many entities before starting to collect results.
      Use with limit for pagination (e.g., offset=5, limit=5 for page 2).

    SPECIALTY DISCOVERY:
    - If specialties is None and search is None, returns a summary of available
      specialties and entity counts instead of entities.

    Returns {"entities": [...], "pagination": {...}} with entity summaries and metadata.
    """
    candidates = normalize_records(candidates) if candidates else []
    network = normalize_records(network) if network else []
    candidates_df = pd.DataFrame(candidates) if candidates else pd.DataFrame()
    network_df = pd.DataFrame(network) if network else pd.DataFrame()

    if candidates_df.empty:
        return {"entities": [], "pagination": {"total_matching": 0, "offset": offset, "limit": limit, "returned": 0}}

    entity_col = "entity" if "entity" in candidates_df.columns else None
    spec_col = "specialty" if "specialty" in candidates_df.columns else None
    city_col = "city" if "city" in candidates_df.columns else None

    # --- Specialty discovery mode ---
    if specialties is None and (search is None or search.strip() == ""):
        if spec_col is None:
            return {"entities": [], "pagination": {"total_matching": 0, "offset": offset, "limit": limit, "returned": 0}, "available_specialties": {}, "total_entities": 0, "hint": "No specialty column found."}
        spec_counts = candidates_df[spec_col].dropna().str.lower().value_counts().to_dict()
        entity_count = candidates_df[entity_col].nunique() if entity_col else len(candidates_df)
        return {
            "entities": [],
            "pagination": {"total_matching": 0, "offset": offset, "limit": limit, "returned": 0},
            "available_specialties": {k: int(v) for k, v in spec_counts.items()},
            "total_entities": int(entity_count),
            "hint": "Pass 'specialties' to get entities, or 'search' to find entities by name.",
        }

    # --- Build eligible provider mask ---
    eligible_mask = pd.Series(True, index=candidates_df.index)

    # Filter by specialties (exact, case-insensitive)
    if specialties:
        target_specialties_lower = [s.lower() for s in specialties]
        if spec_col is None:
            return {"entities": [], "pagination": {"total_matching": 0, "offset": offset, "limit": limit, "returned": 0}, "error": "No specialty column found in candidate data."}
        spec_match = candidates_df[spec_col].str.lower().isin(target_specialties_lower)
        eligible_mask = eligible_mask & spec_match

    # Filter by search text (entity name, specialty, city)
    if search and search.strip():
        search_lower = search.lower()
        text_match = pd.Series(False, index=candidates_df.index)
        if entity_col and pd.api.types.is_string_dtype(candidates_df[entity_col]):
            text_match = text_match | candidates_df[entity_col].str.lower().str.contains(search_lower, na=False)
        if spec_col and pd.api.types.is_string_dtype(candidates_df[spec_col]):
            text_match = text_match | candidates_df[spec_col].str.lower().str.contains(search_lower, na=False)
        if city_col and pd.api.types.is_string_dtype(candidates_df[city_col]):
            text_match = text_match | candidates_df[city_col].str.lower().str.contains(search_lower, na=False)
        eligible_mask = eligible_mask & text_match

    eligible_providers = candidates_df[eligible_mask]
    if eligible_providers.empty:
        if specialties and search:
            return {"entities": [], "pagination": {"total_matching": 0, "offset": offset, "limit": limit, "returned": 0}, "error": f"No available candidates matching specialties {specialties} and search '{search}'."}
        elif specialties:
            return {"entities": [], "pagination": {"total_matching": 0, "offset": offset, "limit": limit, "returned": 0}, "error": f"No available candidates for the requested specialties: {specialties}."}
        else:
            return {"entities": [], "pagination": {"total_matching": 0, "offset": offset, "limit": limit, "returned": 0}, "error": f"No available candidates matching search '{search}'."}

    eligible_entities = set(eligible_providers[entity_col].dropna().unique()) if entity_col else set()

    # Exclude entities already in the network
    if include_not_in_network and not network_df.empty and entity_col in network_df.columns:
        used_entities = set(network_df[entity_col].unique())
        eligible_entities = eligible_entities - used_entities

    # Exclude specific entity IDs
    if exclude_entity_ids:
        eligible_entities = eligible_entities - set(exclude_entity_ids)

    if not eligible_entities:
        return {"entities": [], "pagination": {"total_matching": 0, "offset": offset, "limit": limit, "returned": 0}, "error": "No available entities match the specified filters."}

    # --- Build entity summaries ---
    if entity_summaries:
        summaries_df = pd.DataFrame(entity_summaries)
        entity_key = "entity" if "entity" in summaries_df.columns else summaries_df.columns[0]
    else:
        summaries_df = _aggregate_entities(candidates)
        entity_key = None

    if entity_key and entity_key in summaries_df.columns:
        summaries_df = summaries_df.set_index(entity_key)
        filtered_entities = summaries_df.loc[
            [e for e in eligible_entities if e in summaries_df.index]
        ]
    else:
        filtered_entities = summaries_df.loc[list(eligible_entities)]

    if filtered_entities.empty:
        return {"entities": [], "pagination": {"total_matching": 0, "offset": offset, "limit": limit, "returned": 0}, "error": "No available entities match the specified filters."}

    # --- Apply min thresholds on aggregated metrics ---
    if min_effectiveness is not None:
        if "avg_effectiveness" in filtered_entities.columns:
            filtered_entities = filtered_entities[filtered_entities["avg_effectiveness"] >= min_effectiveness]
        else:
            if spec_col:
                eff_by_entity = eligible_providers.groupby(entity_col)["effectiveness"].mean()
                filtered_entities = filtered_entities[
                    filtered_entities.index.isin(eff_by_entity[eff_by_entity >= min_effectiveness].index)
                ]

    if min_provider_count is not None:
        if "provider_count" in filtered_entities.columns:
            filtered_entities = filtered_entities[filtered_entities["provider_count"] >= min_provider_count]

    if max_geographic_reach is not None:
        if "geographic_reach" in filtered_entities.columns:
            filtered_entities = filtered_entities[filtered_entities["geographic_reach"] <= max_geographic_reach]

    if filtered_entities.empty:
        return {"entities": [], "pagination": {"total_matching": 0, "offset": offset, "limit": limit, "returned": 0}, "error": "No available entities match the specified filters."}

    # --- Sort ---
    if weighted_metrics:
        def calculate_score(row):
            score = 0.0
            for metric, weight in weighted_metrics.items():
                if metric in row:
                    val = row[metric]
                    if pd.isna(val) or val is None:
                        val = 0.0
                    score += float(val) * weight
            return score

        filtered_entities["_score"] = filtered_entities.apply(calculate_score, axis=1)
        filtered_entities = filtered_entities.sort_values(by="_score", ascending=ascending)
    elif sort_by != "none":
        if sort_by in filtered_entities.columns:
            filtered_entities = filtered_entities.sort_values(by=sort_by, ascending=ascending)
        else:
            return {"entities": [], "pagination": {"total_matching": 0, "offset": offset, "limit": limit, "returned": 0}, "error": f"Invalid sort_by value '{sort_by}'. Valid entity metrics are: {list(filtered_entities.columns)}"}

    # --- Paginate ---
    total_count = len(filtered_entities)
    paginated = filtered_entities.iloc[offset:offset + limit]
    if len(paginated) > limit:
        paginated = paginated.head(limit)

    # --- Convert to Entity Summary Objects ---
    results = []
    for entity_id, row in paginated.iterrows():
        results.append({
            "entity_id": entity_id,
            "metrics": {
                "avg_effectiveness": round(row["avg_effectiveness"], 2) if "avg_effectiveness" in row else None,
                "avg_efficiency": round(row["avg_efficiency"], 2) if "avg_efficiency" in row else None,
                "provider_count": int(row["provider_count"]),
                "avg_total_claims": round(row["avg_total_claims_amount"], 2) if "avg_total_claims_amount" in row else None,
                "total_claims": round(row["total_claims_amount"], 2) if "total_claims_amount" in row else None,
                "avg_medicare_claims": round(row["avg_medicare_total_claims_amount"], 2) if "avg_medicare_total_claims_amount" in row else None,
                "total_medicare_claims": round(row["total_medicare_claims_amount"], 2) if "total_medicare_claims_amount" in row else None,
                "new_patient_rate": round(row["new_patient_rate"], 2) if "new_patient_rate" in row else None,
                "geographic_reach": int(row["geographic_reach"]) if "geographic_reach" in row else None,
                "location_confidence": row["location_confidence_dist"] if "location_confidence_dist" in row else None,
            },
            "capabilities": {
                "specialties": row["specialties"] if "specialties" in row else [],
            },
        })

    return {
        "entities": results,
        "pagination": {
            "total_matching": int(total_count),
            "offset": int(offset),
            "limit": int(limit),
            "returned": len(results),
        },
    }


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
    network = normalize_records(network) if network else []
    candidates = normalize_records(candidates) if candidates else []
    network_df = pd.DataFrame(network) if network else pd.DataFrame()
    candidates_df = pd.DataFrame(candidates) if candidates else pd.DataFrame()

    if candidates_df.empty:
        return "No candidate data available."

    entity_col = "entity" if "entity" in candidates_df.columns else None
    
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
    county_specialty_thresholds: Annotated[dict[str, dict[str, dict[str, float]]], InjectedState("county_specialty_thresholds")],
    candidates: Annotated[list[dict], InjectedState("candidates")],
) -> dict[str, Any]:
    """Return the current network status including total providers and member coverage.

    - county_specialty_thresholds: A nested dictionary mapping state->county->specialty
      to threshold dicts (e.g. {"mi": {"wayne": {"general practice": 20.0, "cardiology": 10.0}}}).
      The scope of analysis is defined by the keys present. Default threshold is 20.0 miles.

    Output format:
      {
        "total_providers": int,
        "member_coverage": [
          {"county": str, "specialty": str, "members_with_access": int, "total_members": int, "coverage_percentage": float},
          ...
        ],
        "validation_errors": [str, ...]  # specialties not found in candidate data
      }
    """
    members = normalize_records(members) if members else []
    network = normalize_records(network) if network else []
    candidates = normalize_records(candidates) if candidates else []
    coverage, validation_errors = _compute_coverage(network, members, county_specialty_thresholds, candidates)
    return {
        "total_providers": len(network),
        "member_coverage": coverage,
        "validation_errors": validation_errors,
    }


# ---------------------------------------------------------------------------
# Simulation helpers (used by simulate_network_change)
# ---------------------------------------------------------------------------

def _build_sim_network(
    network_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    add_entity_ids: list[str],
    remove_entity_ids: list[str],
) -> list[dict]:
    """Build a temporary network by applying entity additions and removals."""
    sim_network = [p for _, p in network_df.iterrows()] if not network_df.empty else []
    sim_network = [p.to_dict() if hasattr(p, "to_dict") else p for p in sim_network]

    entity_col = "entity" if "entity" in candidates_df.columns else None

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
    """Compute per-county-and-specialty coverage change between current and simulated."""
    current_map = {(c["county"], c["specialty"]): c["coverage_percentage"] for c in current_coverage}
    sim_map = {(c["county"], c["specialty"]): c["coverage_percentage"] for c in simulated_coverage}
    all_keys = sorted(set(list(current_map.keys()) + list(sim_map.keys())))
    delta = []
    for county, specialty in all_keys:
        change = round(sim_map.get((county, specialty), 0) - current_map.get((county, specialty), 0), 2)
        delta.append({
            "county": county,
            "specialty": specialty,
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

    entity_col = "entity" if "entity" in candidates_df.columns else None
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
    candidates: Annotated[list[dict], InjectedState("candidates")],
    network: Annotated[list[dict], InjectedState("network")],
    members: Annotated[list[dict], InjectedState("members")],
    county_specialty_thresholds: Annotated[dict[str, dict[str, dict[str, float]]], InjectedState("county_specialty_thresholds")],
    add_entity_ids: list[str] = [],
    remove_entity_ids: list[str] = [],
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

    Returns current coverage, simulated coverage, and per-county-specialty delta (percentage point change).
    """
    candidates = normalize_records(candidates) if candidates else []
    network = normalize_records(network) if network else []
    members = normalize_records(members) if members else []
    candidates_df = pd.DataFrame(candidates) if candidates else pd.DataFrame()
    network_df = pd.DataFrame(network) if network else pd.DataFrame()

    current_coverage, current_errors = _compute_coverage(network, members, county_specialty_thresholds, candidates)

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
            sim_cov, sim_errors = _compute_coverage(sim_net, members, county_specialty_thresholds, candidates)
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
    sim_cov, sim_errors = _compute_coverage(sim_net, members, county_specialty_thresholds, candidates)
    delta = _compute_delta(current_coverage, sim_cov)

    all_validation_errors = list(set(current_errors + sim_errors))

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
        "validation_errors": all_validation_errors,
    }


def _run_code_timeout_handler(signum, frame):
    """Signal handler for code execution timeout."""
    raise TimeoutError("Code execution timed out after 20 seconds")


@tool
def run_code(
    code: str,
    candidates: Annotated[list[dict], InjectedState("candidates")],
    entity_summaries: Annotated[list[dict], InjectedState("entity_summaries")],
    network: Annotated[list[dict], InjectedState("network")],
    members: Annotated[list[dict], InjectedState("members")],
    county_specialty_thresholds: Annotated[dict, InjectedState("county_specialty_thresholds")],
) -> str:
    """Execute Python/pandas code to filter, analyze, or simulate network changes.

    The following variables are available in the sandbox:
    - candidates_df: Raw provider-level candidate data (DataFrame)
    - entity_summaries_df: Pre-aggregated entity summaries (DataFrame)
    - network_df: Currently contracted providers (DataFrame)
    - members_df: Member locations (DataFrame)
    - thresholds: Service area configuration (dict)
    - compute_coverage: Helper function to calculate coverage. 
      Usage: compute_coverage(network_df, members_df, thresholds, candidates_df)
      Returns: (coverage_results, validation_errors)

    Assign your result to 'result'. Returns as JSON.
    Allowed: pandas, numpy, json, math, functools, itertools, collections, sklearn.neighbors.BallTree.
    Timeout: 20 seconds.

    Example (Simulation):
      sim_net = pd.concat([network_df, candidates_df[candidates_df['entity'] == 'Covenant']])
      result, errs = compute_coverage(sim_net, members_df, thresholds, candidates_df)
    """
    sandbox_globals = {
        "__builtins__": {},
        "pd": pd,
        "np": np,
        "json": json,
        "math": math,
        "functools": functools,
        "itertools": itertools,
        "defaultdict": defaultdict,
        "candidates_df": pd.DataFrame(candidates) if candidates else pd.DataFrame(),
        "entity_summaries_df": pd.DataFrame(entity_summaries) if entity_summaries else pd.DataFrame(),
        "network_df": pd.DataFrame(network) if network else pd.DataFrame(),
        "members_df": pd.DataFrame(members) if members else pd.DataFrame(),
        "thresholds": county_specialty_thresholds,
        "compute_coverage": compute_coverage_df,
    }

    # Use threading-based timeout since signal.alarm doesn't work in non-main threads
    import threading

    result_holder = {"value": None, "error": None}
    timer = threading.Timer(20.0, lambda: None)  # placeholder

    def _execute():
        try:
            exec(code, sandbox_globals)
            result_holder["value"] = sandbox_globals.get("result")
        except TimeoutError as e:
            result_holder["error"] = f"Timeout: {str(e)}"
        except Exception as e:
            result_holder["error"] = f"{type(e).__name__}: {str(e)}"

    thread = threading.Thread(target=_execute)
    thread.daemon = True
    thread.start()
    thread.join(timeout=20.0)

    if thread.is_alive():
        return "Error: Code execution timed out after 20 seconds."

    if result_holder["error"]:
        return f"Error: {result_holder['error']}"

    result = result_holder["value"]

    if isinstance(result, pd.DataFrame):
        return result.to_dict(orient="records")
    elif isinstance(result, (dict, list, str, int, float, bool, type(None))):
        return result
    else:
        return str(result)


# ---------------------------------------------------------------------------
# Tool collection
# ---------------------------------------------------------------------------

TOOLS = [add_contract_entity, get_network_status, run_code]
