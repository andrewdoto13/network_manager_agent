"""Data loading functions for the network management agent."""

import pandas as pd
from pathlib import Path

from .config import DATA_DIR

# Maps canonical column names to possible CSV column name variants (case-insensitive)
CANONICAL_COLUMNS = {
    "lat": ["latitude", "lat"],
    "lon": ["longitude", "lon"],
    "county": ["county"],
    "entity": ["primary contract entity", "entity"],
    "specialty": ["specialty"],
    "effectiveness": ["effectiveness"],
    "efficiency": ["efficiency"],
    "location_confidence": ["location confidence score"],
    "total_claims_amount": ["total claims amount"],
    "medicare_total_claims_amount": ["medicare total claims amount"],
    "new_patient_claims": ["medicare new patient claims"],
    "claims_volume": ["medicare claims volume", "total claims volume"],
    "city": ["city"],
}


def _resolve_column(df_columns: list[str], canonical: str) -> str | None:
    """Find the actual column name in df_columns that matches the canonical name.
    
    Returns the actual column name if found, None otherwise.
    """
    synonyms = CANONICAL_COLUMNS.get(canonical, [canonical])
    for col in df_columns:
        if col.lower() in [s.lower() for s in synonyms]:
            return col
    return None


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename CSV columns to canonical names in-place.
    
    This ensures all downstream code can reference columns by their canonical
    names without any discovery logic.
    """
    renames = {}
    for canonical, synonyms in CANONICAL_COLUMNS.items():
        actual = _resolve_column(df.columns, canonical)
        if actual is not None and actual != canonical:
            renames[actual] = canonical
    df.rename(columns=renames, inplace=True)
    return df


def normalize_coordinates(df: pd.DataFrame) -> None:
    """Normalize scaled integer coordinates in-place.

    Heuristic: if latitude mean is significantly outside [-90, 90], values are
    likely scaled by 1e6 (e.g. 43451784 -> 43.451784). Longitudes are negated
    to ensure they are negative (West).

    Args:
        df: DataFrame with lat/lon columns to normalize.
    """
    lat_col = _resolve_column(df.columns, "lat")
    lon_col = _resolve_column(df.columns, "lon")

    if lat_col is None or lon_col is None:
        return

    if df[lat_col].abs().mean() > 1000:
        df[lat_col] = df[lat_col] / 1_000_000.0
        df[lon_col] = df[lon_col] / 1_000_000.0
        df.loc[df[lon_col] > 0, lon_col] *= -1


def load_candidates(path: Path | None = None) -> list[dict]:
    """Load candidate entities from CSV and normalize column names and coordinates.
    
    Args:
        path: Path to candidates CSV. Defaults to data/raw/mi_market_data.csv.
    
    Returns:
        List of candidate dicts with canonical column names, normalized lat/lon,
        and an added 'id' column.
    """
    if path is None:
        path = DATA_DIR / "mi_market_data.csv"

    df = pd.read_csv(path).reset_index().rename(columns={"index": "id"})
    _normalize_columns(df)
    normalize_coordinates(df)
    return df.to_dict(orient="records")



def load_members(path: Path | None = None) -> list[dict]:
    """Load member locations from CSV and normalize column names.

    Args:
        path: Path to members.csv. Defaults to data/raw/members.csv.

    Returns:
        List of member dicts with canonical column names and an added 'id' column.
    """
    if path is None:
        path = DATA_DIR / "members.csv"

    df = pd.read_csv(path).reset_index().rename(columns={"index": "id"})
    _normalize_columns(df)
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
        Tuple of (candidates, members) as lists of dicts with canonical column names.
    """
    candidates = load_candidates(candidates_path)
    members = load_members(members_path)
    return candidates, members


def normalize_records(records: list[dict]) -> list[dict]:
    """Normalize column names in a list of record dicts.
    
    Converts dicts to DataFrame, applies canonical column renaming,
    and converts back to dicts. This ensures downstream code always
    sees canonical names regardless of the source.
    
    Args:
        records: List of dicts (possibly with non-canonical column names).
    
    Returns:
        List of dicts with canonical column names.
    """
    if not records:
        return records
    df = pd.DataFrame(records)
    _normalize_columns(df)
    return df.to_dict(orient="records")

