"""Data loading and management for the network management agent."""

import pandas as pd
from pathlib import Path

from .config import DATA_DIR, SERVICE_AREA_BUFFER_MILES

# Maps canonical column names to possible CSV column name variants (case-insensitive)
CANONICAL_COLUMNS = {
    "lat": ["latitude", "lat"],
    "lon": ["longitude", "lon"],
    "state": ["state"],
    "county": ["county", "countyname"],
    "entity": ["primary contract entity", "entity"],
    "specialty": ["specialty"],
    "effectiveness": ["effectiveness"],
    "efficiency": ["efficiency"],
    "location_confidence": ["location confidence score"],
    "total_claims_amount": ["total claims amount"],
    "medicare_total_claims_amount": ["medicare total claims amount"],
    "new_patient_claims": ["medicare new patient claims"],
    "medicare_claims_volume": ["medicare claims volume"],
    "total_claims_volume": ["total claims volume"],
    "city": ["city"],
}


class DataManager:
    """Singleton manager for loading, filtering, and providing access to network data."""
    _instance = None

    # Column name mapping shared across all instances
    CANONICAL_COLUMNS = CANONICAL_COLUMNS

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(DataManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        candidates_path: Path | None = None,
        members_path: Path | None = None,
        county_specialty_thresholds: dict = None
    ):
        if self._initialized:
            return

        self.candidates_path = candidates_path or (DATA_DIR / "mi_market_data.csv")
        self.members_path = members_path or (DATA_DIR / "MedicareSampleCensus2023Q4.csv")
        self.thresholds = county_specialty_thresholds or {}

        self._load_and_filter()
        self._initialized = True

    # -----------------------------------------------------------------------
    # Column resolution & normalization
    # -----------------------------------------------------------------------

    @staticmethod
    def _resolve_column(df_columns: list[str], canonical: str) -> str | None:
        """Find the actual column name in df_columns that matches the canonical name."""
        synonyms = CANONICAL_COLUMNS.get(canonical, [canonical])
        for col in df_columns:
            if col.lower() in [s.lower() for s in synonyms]:
                return col
        return None

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Rename CSV columns to canonical names in-place."""
        renames = {}
        for canonical, synonyms in CANONICAL_COLUMNS.items():
            actual = DataManager._resolve_column(df.columns, canonical)
            if actual is not None and actual != canonical:
                renames[actual] = canonical
        df.rename(columns=renames, inplace=True)
        return df

    @staticmethod
    def _normalize_coordinates(df: pd.DataFrame) -> None:
        """Normalize scaled integer coordinates in-place."""
        lat_col = DataManager._resolve_column(df.columns, "lat")
        lon_col = DataManager._resolve_column(df.columns, "lon")

        if lat_col is None or lon_col is None:
            return

        if df[lat_col].abs().mean() > 1000:
            df[lat_col] = df[lat_col] / 1_000_000.0
            df[lon_col] = df[lon_col] / 1_000_000.0

        # Flip positive longitudes to negative (Western hemisphere)
        if df[lon_col].gt(0).any() and df[lon_col].lt(0).sum() == 0:
            df[lon_col] = -df[lon_col]

    @staticmethod
    def _normalize_strings(df: pd.DataFrame) -> pd.DataFrame:
        """Strip whitespace and lowercase all string columns in-place."""
        for col in df.select_dtypes(include=["object", "string"]).columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
            df[col] = df[col].replace("nan", pd.NA)
        return df

    # -----------------------------------------------------------------------
    # Service area filtering
    # -----------------------------------------------------------------------

    @staticmethod
    def _filter_by_service_area(
        cdf: pd.DataFrame,
        mdf: pd.DataFrame,
        thresholds: dict
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Filter candidates and members to the defined service area."""
        state_counties: dict[str, set[str]] = {}
        all_thresholds: dict[str, dict[str, float]] = {}
        for state_val, counties in thresholds.items():
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
        buffer_deg = (max_threshold + SERVICE_AREA_BUFFER_MILES) / 69.0

        if "state" in mdf.columns:
            mdf = mdf.copy()
            mdf["state_lower"] = mdf["state"].astype(str).str.lower()
            mdf["county_lower"] = mdf["county"].astype(str).str.lower()
            all_target_counties = set().union(*state_counties.values())
            mask = (
                mdf["state_lower"].isin(state_counties.keys())
                & mdf["county_lower"].isin(all_target_counties)
            )
            mdf = mdf[mask].drop(columns=["state_lower", "county_lower"], errors="ignore")
        else:
            all_counties = set()
            for counties in state_counties.values():
                all_counties.update(counties)
            mdf = mdf[mdf["county"].str.lower().isin(all_counties)]

        if mdf.empty:
            return pd.DataFrame(), mdf

        lat_min = mdf["lat"].min() - buffer_deg
        lat_max = mdf["lat"].max() + buffer_deg
        lon_min = mdf["lon"].min() - buffer_deg
        lon_max = mdf["lon"].max() + buffer_deg

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
        return cdf.loc[entity_mask].reset_index(drop=True), mdf.reset_index(drop=True)

    # -----------------------------------------------------------------------
    # Internal loading
    # -----------------------------------------------------------------------

    def _load_and_filter(self):
        """Load raw data and apply service area filtering."""
        cdf_raw = pd.read_csv(self.candidates_path).reset_index().rename(columns={"index": "id"})
        self._normalize_columns(cdf_raw)
        self._normalize_coordinates(cdf_raw)
        self._normalize_strings(cdf_raw)

        mdf_raw = pd.read_csv(self.members_path).reset_index().rename(columns={"index": "id"})
        self._normalize_columns(mdf_raw)
        self._normalize_coordinates(mdf_raw)
        self._normalize_strings(mdf_raw)

        if not self.thresholds:
            self.candidates_df = cdf_raw
            self.members_df = mdf_raw
        else:
            self.candidates_df, self.members_df = self._filter_by_service_area(cdf_raw, mdf_raw, self.thresholds)

    # -----------------------------------------------------------------------
    # Public accessors
    # -----------------------------------------------------------------------
    def get_candidates_df(self) -> pd.DataFrame:
        return self.candidates_df

    def get_members_df(self) -> pd.DataFrame:
        return self.members_df

    def get_providers_by_entity(self, entity_id: str) -> pd.DataFrame:
        """Return all providers for a given entity."""
        return self.candidates_df[self.candidates_df["entity"] == entity_id]

    @classmethod
    def reset(cls):
        """Reset the singleton instance."""
        cls._instance = None
