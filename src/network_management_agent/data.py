"""Data loading functions for the network management agent."""

import pandas as pd
from pathlib import Path


def load_hospitals(path: Path | None = None) -> list[dict]:
    """Load hospital candidates from CSV.

    Args:
        path: Path to hospitals.csv. Defaults to data/raw/hospitals.csv.

    Returns:
        List of hospital dicts with an added 'id' column.
    """
    if path is None:
        path = Path(__file__).parent.parent.parent / "data" / "raw" / "hospitals.csv"

    df = pd.read_csv(path).reset_index().rename(columns={"index": "id"})
    return df.to_dict(orient="records")


def load_members(path: Path | None = None) -> list[dict]:
    """Load member locations from CSV.

    Args:
        path: Path to members.csv. Defaults to data/raw/members.csv.

    Returns:
        List of member dicts with an added 'id' column.
    """
    if path is None:
        path = Path(__file__).parent.parent.parent / "data" / "raw" / "members.csv"

    df = pd.read_csv(path).reset_index().rename(columns={"index": "id"})
    return df.to_dict(orient="records")


def load_data(
    hospitals_path: Path | None = None,
    members_path: Path | None = None,
) -> tuple[list[dict], list[dict]]:
    """Load both hospitals and members data.

    Args:
        hospitals_path: Path to hospitals CSV.
        members_path: Path to members CSV.

    Returns:
        Tuple of (candidates, members) as lists of dicts.
    """
    candidates = load_hospitals(hospitals_path)
    members = load_members(members_path)
    return candidates, members
