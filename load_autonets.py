import pickle
import numpy as np
from pathlib import Path

NETWORKS_DIR = Path("data/networks")


def load_autonets_rs(version):
    """Load revScope autonomous networks (always pruned)."""
    path = NETWORKS_DIR / f"autonets_rs_P_v{version}.pkl"
    with open(path, "rb") as f:
        nets = pickle.load(f)
    print(f"Loaded {len(nets)} autonomous networks from {path}")
    return nets


def load_autonets_np(pruner, paths_version, version, pruned=True):
    """Load NumPaths-derived autonomous networks."""
    suffix = "P" if pruned else "NP"
    path = NETWORKS_DIR / f"autonets_np_{pruner}_{suffix}_pv{paths_version}_v{version}.pkl"
    with open(path, "rb") as f:
        nets = pickle.load(f)
    print(f"Loaded {len(nets)} autonomous networks from {path}")
    return nets


if __name__ == "__main__":
    nets = load_autonets_rs(version=2)
    for i, net in enumerate(nets[:5]):
        print(f"  Network {i}: {len(net)} reactions")
