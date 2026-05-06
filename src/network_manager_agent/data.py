"""Data loading and management for the network management agent."""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any

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
    # Aggregation & profiling
    # -----------------------------------------------------------------------

    @staticmethod
    def aggregate_entities(candidates: pd.DataFrame | list[dict]) -> pd.DataFrame:
        """Aggregate provider-level data into entity-level summaries.

        Args:
            candidates: Provider-level records as a DataFrame or list of dicts.
        """
        if isinstance(candidates, list) and not candidates:
            return pd.DataFrame()
        if isinstance(candidates, pd.DataFrame) and candidates.empty:
            return pd.DataFrame()

        df = pd.DataFrame(candidates)
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
            df = df.copy()
            df["entity"] = df.index
            entity_col = "entity"

        agg_map = {}
        if eff_col: agg_map[eff_col] = "mean"
        if eta_col: agg_map[eta_col] = "mean"
        if spec_col:
            agg_map[spec_col] = (
                lambda x: x.value_counts().head(10).index.tolist()
                + (
                    [f"... and {x.dropna().nunique() - 10} more"]
                    if x.dropna().nunique() > 10
                    else []
                )
            )
        if new_pat_col: agg_map[new_pat_col] = lambda x: float(round((x == 'yes').mean() * 100, 2)) if not x.empty else None
        if medicare_claims_vol_col:
            agg_map[medicare_claims_vol_col] = lambda x: {k: int(v) for k, v in x.dropna().value_counts().to_dict().items()}
        if total_claims_vol_col:
            agg_map[total_claims_vol_col] = lambda x: {k: int(v) for k, v in x.dropna().value_counts().to_dict().items()}
        if city_col: agg_map[city_col] = "nunique"
        agg_map[entity_col] = "count"
        if claims_col: agg_map[claims_col] = lambda x: float(round(x.dropna().mean(), 2)) if x.dropna().any() else None
        if medicare_claims_col: agg_map[medicare_claims_col] = lambda x: float(round(x.dropna().mean(), 2)) if x.dropna().any() else None
        if confidence_col:
            agg_map[confidence_col] = lambda x: {k: int(v) for k, v in x.dropna().value_counts().to_dict().items()}

        agg_df = df.groupby(entity_col).agg(agg_map)

        if claims_col:
            totals = df.groupby(entity_col)[claims_col].apply(lambda x: float(round(x.dropna().sum(), 2)) if x.dropna().any() else None)
            totals.name = "_sum_total_claims_amount"
            agg_df = agg_df.join(totals)
        if medicare_claims_col:
            totals = df.groupby(entity_col)[medicare_claims_col].apply(lambda x: float(round(x.dropna().sum(), 2)) if x.dropna().any() else None)
            totals.name = "_sum_medicare_total_claims_amount"
            agg_df = agg_df.join(totals)

        agg_df = agg_df.rename(columns={
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
        return agg_df

    @staticmethod
    def build_schema_profile(entity_df: pd.DataFrame) -> dict:
        """Build a statistical profile from an entity DataFrame."""
        if entity_df.empty:
            return {}

        profile = {}
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
            else:
                first_val = entity_df[col].dropna().iloc[0] if not entity_df[col].dropna().empty else None
                if isinstance(first_val, (list, dict)):
                    all_vals = [item for sublist in entity_df[col].dropna() for item in (sublist if isinstance(sublist, list) else sublist.keys() if isinstance(sublist, dict) else [sublist])]
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

        self.entity_summaries_df = self.aggregate_entities(self.candidates_df)
        self.entity_summaries = self.entity_summaries_df.reset_index().to_dict(orient="records") if not self.entity_summaries_df.empty else []
        self.schema_profile = self.build_schema_profile(self.entity_summaries_df)
        self.raw_candidate_schema = self.build_schema_profile(self.candidates_df)

    # -----------------------------------------------------------------------
    # Public accessors
    # -----------------------------------------------------------------------

    def get_candidates_df(self) -> pd.DataFrame:
        return self.candidates_df

    def get_members_df(self) -> pd.DataFrame:
        return self.members_df

    def get_entity_summaries(self) -> list[dict]:
        return self.entity_summaries

    def get_schema_profile(self) -> dict:
        return self.schema_profile

    def get_raw_candidate_schema_profile(self) -> dict:
        return self.raw_candidate_schema

    def get_providers_by_entity(self, entity_id: str) -> pd.DataFrame:
        """Return all providers for a given entity."""
        return self.candidates_df[self.candidates_df["entity"] == entity_id]

    @classmethod
    def reset(cls):
        """Reset the singleton instance."""
        cls._instance = None
