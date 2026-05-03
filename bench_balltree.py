"""Benchmark BallTree performance on full vs filtered member sets."""
import time
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

from network_manager_agent.config import DATA_DIR
from network_manager_agent.data import DataManager


def _deg2rad(df: pd.DataFrame) -> np.ndarray:
    return df[["lat", "lon"]].values * (np.pi / 180.0)


def _miles_to_radians(miles: float, earth_radius: float = 3958.8) -> float:
    return miles / earth_radius


def main():
    # --- Load unfiltered data ---
    dm = DataManager(county_specialty_thresholds={})
    candidates_df = dm.get_candidates_df()
    members_df = dm.get_members_df()

    print(f"Candidates: {len(candidates_df):,}")
    print(f"Members:    {len(members_df):,}")
    print()

    # --- Load filtered data (washtenaw + wayne) ---
    DataManager.reset()
    dm2 = DataManager(
        county_specialty_thresholds={
            "MI": {
                "washtenaw": {"cardiology": 10.0, "general practice": 20.0},
                "wayne": {"cardiology": 10.0, "general practice": 20.0, "orthopedic surgery": 15.0},
            }
        }
    )
    filtered_members = dm2.get_members_df()
    print(f"Filtered members (washtenaw + wayne): {len(filtered_members):,}")
    print()

    # --- Top 5 entities by provider count ---
    top5 = candidates_df["entity"].value_counts().head(5)
    print(f"Top 5 entities:\n{top5.to_string()}")
    print()

    threshold_miles = 20.0
    radius_rad = _miles_to_radians(threshold_miles)

    # --- Benchmark: unfiltered members ---
    print("=" * 70)
    print("BENCHMARK: Full member set ({:,} members)".format(len(members_df)))
    print("=" * 70)

    group_pts = _deg2rad(members_df)

    results_full = []
    for entity, provider_count in top5.items():
        entity_df = candidates_df[candidates_df["entity"] == entity]

        # BallTree build
        t0 = time.perf_counter()
        tree = BallTree(_deg2rad(entity_df), leaf_size=40, metric="haversine")
        build_time = time.perf_counter() - t0

        # Query radius
        t0 = time.perf_counter()
        indices = tree.query_radius(group_pts, r=radius_rad, return_distance=False)
        query_time = time.perf_counter() - t0

        covered = int(np.array([len(lst) > 0 for lst in indices]).sum())
        pct = covered / len(members_df) * 100

        results_full.append({
            "entity": entity,
            "providers": provider_count,
            "build_ms": round(build_time * 1000, 2),
            "query_ms": round(query_time * 1000, 2),
            "total_ms": round((build_time + query_time) * 1000, 2),
            "covered": covered,
            "coverage_pct": round(pct, 2),
        })
        print(f"  {entity:<45s} build={build_time:8.3f}s  query={query_time:8.3f}s  covered={covered:,} ({pct:.2f}%)")

    print()

    # --- Benchmark: filtered members ---
    print("=" * 70)
    print("BENCHMARK: Filtered member set ({:,} members)".format(len(filtered_members)))
    print("=" * 70)

    filtered_pts = _deg2rad(filtered_members)

    results_filtered = []
    for entity, provider_count in top5.items():
        entity_df = candidates_df[candidates_df["entity"] == entity]

        t0 = time.perf_counter()
        tree = BallTree(_deg2rad(entity_df), leaf_size=40, metric="haversine")
        build_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        indices = tree.query_radius(filtered_pts, r=radius_rad, return_distance=False)
        query_time = time.perf_counter() - t0

        covered = int(np.array([len(lst) > 0 for lst in indices]).sum())
        pct = covered / len(filtered_members) * 100

        results_filtered.append({
            "entity": entity,
            "providers": provider_count,
            "build_ms": round(build_time * 1000, 2),
            "query_ms": round(query_time * 1000, 2),
            "total_ms": round((build_time + query_time) * 1000, 2),
            "covered": covered,
            "coverage_pct": round(pct, 2),
        })
        print(f"  {entity:<45s} build={build_time:8.3f}s  query={query_time:8.3f}s  covered={covered:,} ({pct:.2f}%)")

    print()

    # --- Summary ---
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Full member query times:    {[r['query_ms'] for r in results_full]} ms")
    print(f"  Filtered member query times: {[r['query_ms'] for r in results_filtered]} ms")
    full_avg = np.mean([r["query_ms"] for r in results_full])
    filt_avg = np.mean([r["query_ms"] for r in results_filtered])
    print(f"  Average query (full):       {full_avg:.1f} ms")
    print(f"  Average query (filtered):   {filt_avg:.1f} ms")
    if filt_avg > 0:
        print(f"  Speedup:                      {full_avg / filt_avg:.1f}x")


if __name__ == "__main__":
    main()
