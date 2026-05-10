"""Tool definitions for the network management agent."""

import json
import math

import collections
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

from .data import DataManager


def _blocked_import(name: str, *args, **kwargs):
    raise ImportError(
        "import is disabled. Use pre-injected modules: "
        "pd (pandas), np (numpy), json, math, BallTree, collections. "
        "Do NOT write 'import' statements."
    )


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

def compute_coverage(
    network_input,
    members_df: pd.DataFrame,
    county_specialty_thresholds: dict[str, dict[str, dict[str, float]]],
    candidates_df: pd.DataFrame = None,
) -> tuple[list[dict], list[str]]:
    """Compute per-county-and-specialty member coverage.

    Accepts network_input as one of:
    - list[str]: entity IDs, hydrated via DataManager
    - list[dict]: provider records, converted to DataFrame
    - pd.DataFrame: used directly
    - empty/None: returns empty coverage

    Thresholds use nested structure: {"mi": {"wayne": {"general practice": 20.0}}}.
    """
    if isinstance(network_input, pd.DataFrame):
        net_df = network_input
    elif isinstance(network_input, list):
        if not network_input:
            net_df = pd.DataFrame()
        elif isinstance(network_input[0], str):
            dm = DataManager()
            candidates = dm.get_candidates_df()
            net_df = candidates[candidates["entity"].str.lower().isin([e.lower() for e in network_input])]
        else:
            net_df = pd.DataFrame(network_input)
    else:
        net_df = pd.DataFrame()

    if members_df.empty:
        return [], []

    if "county" not in members_df.columns:
        return [], ["Required column 'county' not found in members data."]

    spec_col = None
    valid_specialties_lower = set()
    if candidates_df is not None and not candidates_df.empty:
        spec_col = "specialty" if "specialty" in candidates_df.columns else None
        if spec_col:
            valid_specialties_lower = set(candidates_df[spec_col].dropna().str.lower().unique().tolist())

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
            county_mask = members_df["county"].astype(str).str.lower() == county_val.lower()
            if "state" in members_df.columns:
                state_mask = members_df["state"].astype(str).str.lower() == state_val.lower()
                county_mask = county_mask & state_mask
            county_members = members_df[county_mask]

            if county_members.empty:
                for specialty, threshold in specialties.items():
                    coverage_results.append({
                        "state": state_val,
                        "county": county_val,
                        "specialty": specialty,
                        "members_with_access": 0,
                        "total_members": 0,
                        "coverage_percentage": 0.0,
                    })
                continue

            group_pts = _deg2rad(county_members)

            for specialty, threshold in specialties.items():
                if spec_col and "specialty" in net_df.columns:
                    specialty_network = net_df[net_df[spec_col].str.lower() == specialty.lower()]
                else:
                    specialty_network = pd.DataFrame()

                if specialty_network.empty:
                    coverage_results.append({
                        "state": state_val,
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
                    "state": state_val,
                    "county": county_val,
                    "specialty": specialty,
                    "members_with_access": members_with_access,
                    "total_members": total_members,
                    "coverage_percentage": coverage_percentage,
                })

    coverage_results.sort(key=lambda x: (x.get("state", ""), x["county"], x["specialty"]))
    return coverage_results, validation_errors


# ---------------------------------------------------------------------------
# LangGraph Tools
# ---------------------------------------------------------------------------

@tool
def add_contract_entity(
    entity_ids: list[str],
    network: Annotated[list[str], InjectedState("network")],
):
    """Add one or more contract entities to the network. Use ONLY after analysis is complete and you have decided which entities to include.

    Parameters:
      entity_ids: List of entity names to add (e.g., ["Covenant Healthcare", "Beaumont Health"]).

    Matching is case-insensitive against the entity names in candidates_df. Returns a dict with:
      - added_entities: List of successfully added canonical entity names
      - skipped_entities: List of entities already in the network
      - errors: List of entity names not found in candidates

    To see the current network state, use run_code to inspect network_df."""
    dm = DataManager()
    candidates_df = dm.get_candidates_df()

    if candidates_df.empty:
        return "No candidate data available."

    entity_col = "entity" if "entity" in candidates_df.columns else None
    used_entities_lower = {e.lower() for e in network}
    added_entities: list[str] = []
    skipped_entities: list[str] = []
    errors: list[str] = []

    for eid in entity_ids:
        if eid.lower() in used_entities_lower:
            skipped_entities.append(eid)
            continue

        match = candidates_df[candidates_df[entity_col] == eid.lower()] if entity_col in candidates_df.columns else pd.DataFrame()
        if match.empty:
            errors.append(f"Entity {eid} not found in candidates.")
        else:
            canonical_name = match.iloc[0][entity_col]
            added_entities.append(canonical_name)

    result: dict[str, Any] = {"added_entities": added_entities}
    if skipped_entities:
        result["skipped_entities"] = skipped_entities
    if errors:
        result["errors"] = errors
    return result


@tool
def run_code(
    network: Annotated[list[str], InjectedState("network")],
    county_specialty_thresholds: Annotated[dict, InjectedState("county_specialty_thresholds")],
    code: str,
):
    """Execute Python/pandas code to filter, analyze, or simulate network changes.

    **IMPORTANT**: Each call is a completely fresh sandbox — variables from previous calls are NOT available. Do NOT write import statements.

    Available variables:
      - candidates_df: Provider-level candidate data. Columns: entity, specialty, lat, lon, effectiveness, efficiency, new_patient_claims, ...
      - network_df: Currently contracted providers (filtered from candidates_df by the network state).
      - members_df: Member locations. Columns: state, county, lat, lon, ...
      - thresholds: Service area config dict, e.g. {"mi": {"wayne": {"general practice": 20.0}}}
      - compute_coverage: See below.

    compute_coverage(network_df, members_df, thresholds, candidates_df) → (coverage_list, errors_list)
      Computes per-county-and-specialty member coverage using haversine distance.
      coverage_list: list of dicts with {state, county, specialty, members_with_access, total_members, coverage_percentage}
      errors_list: list of validation error strings (check this for issues).

    Assign your result to 'result' (must be a JSON-serializable variable). Timeout: 60 seconds.
    Tip: Use BallTree for haversine distance queries; use compute_coverage() for coverage simulation.
    """
    dm = DataManager()

    candidates = dm.get_candidates_df()
    if network:
        net_df = candidates[candidates["entity"].str.lower().isin([e.lower() for e in network])]
    else:
        net_df = candidates.iloc[:0].copy()

    sandbox_globals = {
        "__builtins__": {
            "len": len,
            "sorted": sorted,
            "range": range,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "set": set,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "isinstance": isinstance,
            "type": type,
            "print": print,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "any": any,
            "all": all,
            "None": None,
            "True": True,
            "False": False,
            "ValueError": ValueError,
            "KeyError": KeyError,
            "TypeError": TypeError,
            "IndexError": IndexError,
            "AttributeError": AttributeError,
            "Exception": Exception,
            "__import__": _blocked_import,
        },
        "pd": pd,
        "np": np,
        "json": json,
        "math": math,
        "functools": functools,
        "itertools": itertools,
        "collections": collections,
        "defaultdict": defaultdict,
        "BallTree": BallTree,
        "candidates_df": dm.get_candidates_df(),
        "network_df": net_df,
        "members_df": dm.get_members_df(),
        "thresholds": county_specialty_thresholds,
        "compute_coverage": compute_coverage,
    }

    import threading
    import io
    import sys
    result_holder = {"value": None, "stdout": "", "error": None}

    def _execute():
        try:
            old_stdout = sys.stdout
            sys.stdout = captured = io.StringIO()
            exec(code, sandbox_globals)
            result_holder["stdout"] = captured.getvalue()
            sys.stdout = old_stdout
            result_holder["value"] = sandbox_globals.get("result")
        except Exception as e:
            result_holder["error"] = f"{type(e).__name__}: {str(e)}"
        finally:
            sys.stdout = old_stdout

    thread = threading.Thread(target=_execute)
    thread.daemon = True
    thread.start()
    thread.join(timeout=60.0)

    if thread.is_alive():
        return "Error: Code execution timed out after 60 seconds."

    if result_holder["error"]:
        return f"Error: {result_holder['error']}"

    result = result_holder["value"]
    stdout = result_holder["stdout"].strip()

    # Truncate stdout to prevent context flooding
    _STDOUT_MAX = 2000
    if len(stdout) > _STDOUT_MAX:
        stdout = stdout[:_STDOUT_MAX] + f"\n... [output truncated, {len(stdout) - _STDOUT_MAX} more chars]"

    if isinstance(result, pd.DataFrame):
        output = result.to_dict(orient="records")
    elif isinstance(result, (dict, list, str, int, float, bool, type(None))):
        output = result
    else:
        output = str(result)

    if stdout:
        return f"[stdout]\n{stdout}\n[/stdout]\n{output}" if output else f"[stdout]\n{stdout}\n[/stdout]"
    return output


TOOLS = [add_contract_entity, run_code]
