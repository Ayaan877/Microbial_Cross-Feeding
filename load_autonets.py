import pickle
import numpy as np
from pathlib import Path


def load_autonets(mode, dataset, pruned=True):
    """
    Load autonomous networks from pickle files.
    """
    suffix = "P" if pruned else "NP"
    dir_name = f"AutoNets{dataset}_{mode}_{suffix}"
    dir_path = Path(dir_name)

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    pkl_files = sorted(dir_path.glob("*.pkl"))
    # Filter out yield files
    pkl_files = [f for f in pkl_files if "Yield" not in f.name]

    if not pkl_files:
        raise FileNotFoundError(f"No AutoNet pickle files found in {dir_path}")

    all_nets = []
    for pkl_file in pkl_files:
        with open(pkl_file, "rb") as f:
            nets = pickle.load(f)
        if isinstance(nets, list):
            all_nets.extend(nets)
        else:
            all_nets.append(nets)

    print(f"Loaded {len(all_nets)} autonomous networks from {dir_path}")
    return all_nets


if __name__ == "__main__":
    nets = load_autonets(mode="Batch", dataset=3, pruned=False)
    for i, net in enumerate(nets[:5]):
        print(f"  Network {i}: {len(net)} reactions")
