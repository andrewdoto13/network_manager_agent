"""Data loading functions for the network management agent."""

import pandas as pd
from pathlib import Path

from .config import DATA_DIR


def normalize_coordinates(df: pd.DataFrame) -> None:
    """Normalize scaled integer coordinates in-place.

    Heuristic: if latitude mean is significantly outside [-90, 90], values are
    likely scaled by 1e6 (e.g. 43451784 -> 43.451784). Longitudes are negated
    to ensure they are negative (West).

    Args:
        df: DataFrame with lat/lon columns to normalize.
    """
    lat_col = next((c for c in df.columns if c.lower() in ["lat", "latitude"]), None)
    lon_col = next((c for c in df.columns if c.lower() in ["lon", "longitude"]), None)

    if lat_col is None or lon_col is None:
        return

    if df[lat_col].abs().mean() > 1000:
        df[lat_col] = df[lat_col] / 1_000_000.0
        df[lon_col] = df[lon_col] / 1_000_000.0
        df.loc[df[lon_col] > 0, lon_col] *= -1


def load_candidates(path: Path | None = None) -> list[dict]:
    """Load candidate entities from CSV and normalize coordinates.
    
    Args:
        path: Path to candidates CSV. Defaults to data/raw/mi_market_data.csv.
    
    Returns:
        List of candidate dicts with an added 'id' column and normalized lat/lon.
    """
    if path is None:
        path = DATA_DIR / "mi_market_data.csv"

    df = pd.read_csv(path).reset_index().rename(columns={"index": "id"})
    normalize_coordinates(df)
    return df.to_dict(orient="records")



def load_members(path: Path | None = None) -> list[dict]:
    """Load member locations from CSV.

    Args:
        path: Path to members.csv. Defaults to data/raw/members.csv.

    Returns:
        List of member dicts with an added 'id' column.
    """
    if path is None:
        path = DATA_DIR / "members.csv"

    df = pd.read_csv(path).reset_index().rename(columns={"index": "id"})
    return df.to_dict(orient="records")


def load_data(
    candidates_path: Path | None = None,
    members_path: Path | None = None,
) -> tuple[list[dict], list[dict]]:
    """Load both candidates and members data.
    
    Args:
        candidates_path: Path to candidates CSV.
        members_path: Path to members CSV.
    
    Returns:
        Tuple of (candidates, members) as lists of dicts.
    """
    candidates = load_candidates(candidates_path)
    members = load_members(members_path)
    return candidates, members

