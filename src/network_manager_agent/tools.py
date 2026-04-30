"""Tool definitions for the network management agent."""

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from typing import Annotated

from .data import normalize_coordinates, normalize_records


def _deg2rad(df: pd.DataFrame) -> np.ndarray:
    """Convert lat/lon from degrees to radians."""
    return df[["lat", "lon"]].values * (np.pi / 180.0)


def _miles_to_radians(threshold_miles: float, earth_radius_miles: float = 3958.8) -> float:
    """Convert miles to radians for haversine distance."""
    return threshold_miles / earth_radius_miles


def _compute_coverage(
    network: list[dict],
    members: list[dict],
    county_specialty_thresholds: dict[str, dict[str, float]],
    candidates: list[dict] = None,
) -> tuple[list[dict], list[str]]:
    """Compute per-county-and-specialty member coverage given a network and member set."""
    try:
        members_df = pd.DataFrame(members) if members else pd.DataFrame()
        net_df = pd.DataFrame(network) if network else pd.DataFrame()
        candidates_df = pd.DataFrame(candidates) if candidates else pd.DataFrame()
    except Exception as e:
        return [], [f"Data preparation error: {str(e)}"]
    
    if members_df.empty:
        return [], []
    
    if "county" not in members_df.columns:
        return [], ["Required column 'county' not found in members data."]

    # Discover specialty column in candidates
    spec_col = None
    valid_specialties = set()
    if not candidates_df.empty:
        spec_col = "specialty" if "specialty" in candidates_df.columns else None
        if spec_col:
            valid_specialties = set(candidates_df[spec_col].dropna().unique().tolist())

    # Validate specialties against candidate data
    validation_errors = []
    for county_val, specialties in county_specialty_thresholds.items():
        for spec in specialties.keys():
            if valid_specialties and spec not in valid_specialties:
                validation_errors.append(
                    f"Specialty '{spec}' in county '{county_val}' not found in candidate data."
                )

    coverage_results = []
    for county_val, specialties in county_specialty_thresholds.items():
        county_members = members_df[members_df["county"] == county_val]
        if county_members.empty:
            continue
        
        group_pts = _deg2rad(county_members)

        for specialty, threshold in specialties.items():
            # Filter network providers by specialty

            if spec_col:
                specialty_network = net_df[net_df[spec_col] == specialty]
            else:
                specialty_network = pd.DataFrame()

            if specialty_network.empty:
                coverage_results.append({
                    "county": county_val,
                    "specialty": specialty,
                    "members_with_access": 0,
                    "total_members": len(county_members),
                    "coverage_percentage": 0.0,
                })
                continue

            radius_rad = _miles_to_radians(threshold)
            tree = BallTree(_deg2rad(specialty_network), leaf_size=40, metric="haversine")
            indices, _ = tree.query_radius(group_pts, r=radius_rad, return_distance=True)

            members_with_access = int(np.array([len(lst) > 0 for lst in indices]).sum())
            total_members = len(county_members)
            coverage_percentage = round(members_with_access / total_members * 100, 2)

            coverage_results.append({
                "county": county_val,
                "specialty": specialty,
                "members_with_access": members_with_access,
                "total_members": total_members,
                "coverage_percentage": coverage_percentage,
            })

    coverage_results.sort(key=lambda x: (x["county"], x["specialty"]))
    return coverage_results, validation_errors



def _filter_by_service_area(
    candidates: list[dict],
    members: list[dict],
    county_specialty_thresholds: dict[str, dict[str, float]],
) -> list[dict]:
    """Filter candidates to entities within the service area."""
    candidates = normalize_records(candidates) if candidates else []
    members = normalize_records(members) if members else []
    if not candidates or not members:
        return candidates
 

    if not county_specialty_thresholds:
        return candidates

    max_threshold = max(
        thresh
        for specs in county_specialty_thresholds.values()
        for thresh in specs.values()
    )
    buffer_deg = (max_threshold + 20) / 69.0

    members_df = pd.DataFrame(members)
 
    lat_min = members_df["lat"].min() - buffer_deg
    lat_max = members_df["lat"].max() + buffer_deg
    lon_min = members_df["lon"].min() - buffer_deg
    lon_max = members_df["lon"].max() + buffer_deg
 
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
    result_df = cdf.loc[entity_mask].reset_index(drop=True)

    return result_df.to_dict(orient="records")


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


def precompute_schema_profile(entity_summaries: list[dict]) -> str:
    """Pre-compute the JSON schema profile string from entity summaries.

    Returns a JSON string ready for injection into the system prompt.
    """
    if not entity_summaries:
        return "No schema available."
    df = pd.DataFrame(entity_summaries)
    profile = _build_schema_profile_from_summaries(df)
    return json.dumps(profile, indent=2)


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
    claims_vol_col = "claims_volume" if "claims_volume" in df.columns else None
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
    if claims_vol_col:
        def vol_dist(x):
            dist = x.dropna().value_counts().to_dict()
            return {k: int(v) for k, v in dist.items()}
        agg_map[claims_vol_col] = vol_dist
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
        claims_vol_col: "claims_volume_dist" if claims_vol_col else None,
        city_col: "geographic_reach" if city_col else None,
        "_sum_total_claims_amount": "total_claims_amount",
        "_sum_medicare_total_claims_amount": "total_medicare_claims_amount",
    })


@tool
def get_candidates(
    candidates: Annotated[list[dict], InjectedState("candidates")],
    network: Annotated[list[dict], InjectedState("network")],
    entity_summaries: Annotated[list[dict], InjectedState("entity_summaries")],
    specialties: list[str] = None,
    sort_by: str = "none",
    ascending: bool = True,
    limit: int = 5,
    weighted_metrics: dict[str, float] = None,
):
    """Return up to [limit] contract entities that have at least one provider with the requested specialties and are not yet in the network.

    - sort_by: IMPORTANT - sorts the ENTITIES based on their aggregated metrics (e.g., 'avg_effectiveness', 'provider_count', 'new_patient_rate', 'total_claims_amount', 'avg_total_claims_amount').
      Use 'total_claims_amount' to rank by total entity claims volume. Use 'avg_total_claims_amount' to rank by per-provider average.
      To prioritize high-quality groups, use sort_by='avg_effectiveness' with ascending=False.
      To prioritize accessibility, use sort_by='new_patient_rate' with ascending=False.
    - ascending: if True, sort lowest-first; if False, sort highest-first.
    - weighted_metrics: A dictionary of {metric_name: weight} to create a custom balanced score.
      Example: {"avg_effectiveness": 0.7, "new_patient_rate": 0.3}.
    - limit: The maximum number of entities to return.
    Returns a message if no entities remain for the requested specialties.
    """
    candidates = normalize_records(candidates) if candidates else []
    network = normalize_records(network) if network else []
    candidates_df = pd.DataFrame(candidates) if candidates else pd.DataFrame()
    network_df = pd.DataFrame(network) if network else pd.DataFrame()

    if candidates_df.empty:
        return "No available candidates."

    if specialties is None:
        return "No specialties provided."

    target_specialties = specialties

    spec_col = "specialty" if "specialty" in candidates_df.columns else None
    entity_col = "entity" if "entity" in candidates_df.columns else None

    if spec_col is None:
        return "No specialty column found in candidate data."

    # 1. Find entities that have the specialty
    eligible_providers = candidates_df[candidates_df[spec_col].isin(target_specialties)]
    if eligible_providers.empty:
        return f"No available candidates for the requested specialties: {target_specialties}."

    eligible_entities = set(eligible_providers[entity_col].dropna().unique())

    # 2. Exclude entities that are already (partially) in the network
    if not network_df.empty and entity_col in network_df.columns:
        used_entities = set(network_df[entity_col].unique())
        eligible_entities = eligible_entities - used_entities

    if not eligible_entities:
        return "No available entities for these specialties."

    # 3. Use pre-computed entity summaries (fall back to computing if not provided)
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
            return f"Invalid sort_by value '{sort_by}'. Valid entity metrics are: {list(filtered_entities.columns)}"

    if len(filtered_entities) > limit:
        filtered_entities = filtered_entities.head(limit)

    # 4. Convert to Entity Summary Objects
    results = []
    for entity_id, row in filtered_entities.iterrows():
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

    return results



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
        return pd.DataFrame()

    if entity_df.empty:
        return {}

    return _build_schema_profile_from_summaries(entity_df)

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
    county_specialty_thresholds: Annotated[dict[str, dict[str, float]], InjectedState("county_specialty_thresholds")],
    candidates: Annotated[list[dict], InjectedState("candidates")],
) -> dict[str, Any]:
    """Return the current network status including total providers and member coverage.

    - county_specialty_thresholds: A nested dictionary mapping county names to specialty
      -> threshold dicts (e.g. {"wayne": {"cardiologist": 10.0, "pcp": 5.0}}).
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
    county_specialty_thresholds: Annotated[dict[str, dict[str, float]], InjectedState("county_specialty_thresholds")],
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



TOOLS = [get_candidates, add_contract_entity, get_network_status, simulate_network_change]
