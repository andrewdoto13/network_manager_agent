"""Tool definitions for the network management agent."""

import json
import math

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
            net_df = dm.get_candidates_df()[dm.get_candidates_df()["entity"].isin(network_input)]
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
            group_pts = _deg2rad(members_df)

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


# ---------------------------------------------------------------------------
# LangGraph Tools
# ---------------------------------------------------------------------------

@tool
def add_contract_entity(
    entity_ids: list[str],
    network: Annotated[list[str], InjectedState("network")],
):
    """Add one or more contract entities to the network using their names.

    - entity_ids: List of entity names to add (e.g., ["Covenant Healthcare"]).
    Returns a list of entity IDs added to the network. Entities already in the network
    are silently skipped. Invalid entity names are reported in the 'errors' field.
    """
    dm = DataManager()
    candidates_df = dm.get_candidates_df()

    if candidates_df.empty:
        return "No candidate data available."

    entity_col = "entity" if "entity" in candidates_df.columns else None
    used_entities = set(network)
    added_entities: list[str] = []
    errors: list[str] = []

    for eid in entity_ids:
        if eid in used_entities:
            continue

        match = candidates_df[candidates_df[entity_col] == eid] if entity_col in candidates_df.columns else pd.DataFrame()
        if match.empty:
            errors.append(f"Entity {eid} not found in candidates.")
        else:
            added_entities.append(eid)

    result: dict[str, Any] = {"added_entities": added_entities}
    if errors:
        result["errors"] = errors
    return result


@tool
def run_code(
    network: Annotated[list[str], InjectedState("network")],
    entity_summaries: Annotated[list[dict], InjectedState("entity_summaries")],
    county_specialty_thresholds: Annotated[dict, InjectedState("county_specialty_thresholds")],
    code: str,
):
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
    Timeout: 60 seconds.
    """
    dm = DataManager()

    net_df = dm.get_candidates_df()[dm.get_candidates_df()["entity"].isin(network)] if network else pd.DataFrame()

    sandbox_globals = {
        "__builtins__": {
            "__import__": __import__,
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
        },
        "pd": pd,
        "np": np,
        "json": json,
        "math": math,
        "functools": functools,
        "itertools": itertools,
        "defaultdict": defaultdict,
        "candidates_df": dm.get_candidates_df(),
        "entity_summaries_df": pd.DataFrame(entity_summaries) if entity_summaries else pd.DataFrame(),
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
        except TimeoutError as e:
            result_holder["error"] = f"Timeout: {str(e)}"
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
