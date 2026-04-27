"""Synthetic data generation for the network management agent.

Generates synthetic member locations and hospital candidates clustered
around Wayne County, MI neighborhoods.

Usage:
    python -m scripts.generate_data
    python -m scripts.generate_data --n-members 3000 --n-hospitals 30 --seed 42 --plot
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Bounding box for Wayne County, MI
LAT_MIN, LAT_MAX = 42.00, 42.45
LON_MIN, LON_MAX = -83.60, -82.90

# Cluster centers (lat, lon)
CLUSTER_CENTERS = {
    "detroit": (42.3314, -83.0458),
    "dearborn": (42.3223, -83.1763),
    "livonia": (42.3684, -83.3527),
    "canton": (42.3086, -83.4822),
}


def generate_members(
    n: int = 2000,
    cluster_weight: float = 0.70,
    std_dev: float = 0.02,
    county: str = "wayne",
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Generate synthetic member locations.

    Args:
        n: Total number of member points.
        cluster_weight: Fraction of points clustered around centers.
        std_dev: Standard deviation for cluster point distribution.
        county: County name for all points.
        rng: NumPy random generator for reproducibility.

    Returns:
        DataFrame with lat, lon, county columns.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_cluster = int(n * cluster_weight)
    n_uniform = n - n_cluster

    # Uniform background
    lats_uniform = rng.uniform(LAT_MIN, LAT_MAX, n_uniform)
    lons_uniform = rng.uniform(LON_MIN, LON_MAX, n_uniform)

    # Clustered points
    cluster_names = list(CLUSTER_CENTERS.keys())
    chosen_clusters = rng.choice(cluster_names, size=n_cluster, replace=True)

    lats_cluster, lons_cluster = [], []
    for name in chosen_clusters:
        center_lat, center_lon = CLUSTER_CENTERS[name]
        lats_cluster.append(rng.normal(center_lat, std_dev))
        lons_cluster.append(rng.normal(center_lon, std_dev))

    return pd.DataFrame({
        "lat": np.concatenate([lats_uniform, lats_cluster]),
        "lon": np.concatenate([lons_uniform, lons_cluster]),
        "county": county,
    })


def generate_hospitals(
    n: int = 20,
    mode: str = "mixed",
    std_dev: float = 0.02,
    cluster_weight: float = 0.70,
    county: str = "wayne",
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Generate synthetic hospital candidate locations.

    Args:
        n: Total number of hospital candidates.
        mode: 'cluster' (all clustered) or 'mixed' (clustered + uniform).
        std_dev: Standard deviation for cluster point distribution.
        cluster_weight: Fraction of points clustered (used only in 'mixed' mode).
        county: County name for all hospitals.
        rng: NumPy random generator for reproducibility.

    Returns:
        DataFrame with lat, lon, cluster, county, specialty, effectiveness columns.
    """
    if rng is None:
        rng = np.random.default_rng()

    cluster_names = list(CLUSTER_CENTERS.keys())

    if mode == "cluster":
        chosen = rng.choice(cluster_names, size=n, replace=True)
        lats, lons = [], []
        for name in chosen:
            lat0, lon0 = CLUSTER_CENTERS[name]
            lats.append(rng.normal(lat0, std_dev))
            lons.append(rng.normal(lon0, std_dev))

        return pd.DataFrame({
            "lat": lats,
            "lon": lons,
            "cluster": chosen,
            "county": county,
            "specialty": "hospital",
            "effectiveness": rng.integers(1, 6, size=n),
        })

    elif mode == "mixed":
        n_cluster = int(n * cluster_weight)
        n_uniform = n - n_cluster

        # Clustered portion
        chosen = rng.choice(cluster_names, size=n_cluster, replace=True)
        lats_c, lons_c = [], []
        for name in chosen:
            lat0, lon0 = CLUSTER_CENTERS[name]
            lats_c.append(rng.normal(lat0, std_dev))
            lons_c.append(rng.normal(lon0, std_dev))

        df_cluster = pd.DataFrame({
            "lat": lats_c,
            "lon": lons_c,
            "cluster": chosen,
            "county": county,
            "specialty": "hospital",
            "effectiveness": rng.integers(1, 6, size=n_cluster),
        })

        # Uniform portion
        df_uniform = pd.DataFrame({
            "lat": rng.uniform(LAT_MIN, LAT_MAX, n_uniform),
            "lon": rng.uniform(LON_MIN, LON_MAX, n_uniform),
            "cluster": ["uniform"] * n_uniform,
            "county": county,
            "specialty": "hospital",
            "effectiveness": rng.integers(1, 6, size=n_uniform),
        })

        return pd.concat([df_cluster, df_uniform], ignore_index=True)

    else:
        raise ValueError("mode must be 'cluster' or 'mixed'")


def plot_data(members_df: pd.DataFrame, hospitals_df: pd.DataFrame) -> None:
    """Plot member locations and hospital candidates.

    Args:
        members_df: DataFrame of member locations.
        hospitals_df: DataFrame of hospital candidates.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(6, 6))

    sns.scatterplot(
        x="lon",
        y="lat",
        data=members_df,
        s=10,
        color="blue",
        alpha=0.7,
        edgecolor=None,
    )

    sns.scatterplot(
        x="lon",
        y="lat",
        data=hospitals_df,
        s=120,
        color="crimson",
        edgecolor="black",
        alpha=0.9,
    )

    plt.suptitle("Synthetic Members + Hospital Locations")
    plt.title("Wayne County, MI")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def main() -> None:
    """CLI entry point for synthetic data generation."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic healthcare member and hospital data."
    )
    parser.add_argument(
        "--n-members", type=int, default=2000,
        help="Number of member locations to generate (default: 2000)"
    )
    parser.add_argument(
        "--n-hospitals", type=int, default=20,
        help="Number of hospital candidates to generate (default: 20)"
    )
    parser.add_argument(
        "--mode", choices=["cluster", "mixed"], default="mixed",
        help="Hospital generation mode: 'cluster' or 'mixed' (default: mixed)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/raw"),
        help="Output directory for CSV files (default: data/raw)"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Display visualization plot"
    )

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"Generating {args.n_members} member locations...")
    members_df = generate_members(n=args.n_members, rng=rng)

    print(f"Generating {args.n_hospitals} hospital candidates (mode={args.mode})...")
    hospitals_df = generate_hospitals(n=args.n_hospitals, mode=args.mode, rng=rng)

    args.output.mkdir(parents=True, exist_ok=True)

    members_path = args.output / "members.csv"
    hospitals_path = args.output / "hospitals.csv"

    members_df.to_csv(members_path, index=False)
    hospitals_df.to_csv(hospitals_path, index=False)

    print(f"Saved members to {members_path}")
    print(f"Saved hospitals to {hospitals_path}")

    if args.plot:
        plot_data(members_df, hospitals_df)


if __name__ == "__main__":
    main()
