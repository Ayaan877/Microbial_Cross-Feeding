import pickle
import numpy as np

CORE_METS = [
    "C00009", "C00013", "C00022", "C00025",
    "C00041", "C00065", "C00097", "C00117"]


def loadNumPaths(mode, dataset):
    all_targets = []

    for met in CORE_METS:
        path = f"data/paths/paths_{mode}_{met}_v{dataset}.pkl"
        with open(path, "rb") as f:
            results = pickle.load(f)
            all_targets.append(list(results['networks']))

    return all_targets

if __name__ == "__main__":
    mode = "batch"
    dataset = 3
    all_paths = loadNumPaths(mode=mode, dataset=dataset)

    print(f'Paths: data/paths/ | pruner={mode} | version={dataset}')
    for i, met in enumerate(CORE_METS):
        print('------------------------------')
        print(f"Target {i+1} ({met}): {len(all_paths[i])} pathways")
        sizes = [len(p) for p in all_paths[i]]
        print(f"  Size range: {min(sizes)}-{max(sizes)}, mean: {np.mean(sizes):.1f}")