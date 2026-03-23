import pickle
import numpy as np

def loadNumPaths(mode, dataset):
    target_files = [
            "C00009_Pathways.pkl",
            "C00013_Pathways.pkl",
            "C00022_Pathways.pkl",
            "C00025_Pathways.pkl",
            "C00041_Pathways.pkl",
            "C00065_Pathways.pkl",
            "C00097_Pathways.pkl",
            "C00117_Pathways.pkl"]
    
    if mode == "batch":
        data_dir = f"{dataset}_Batch"
    elif mode == "single":
        data_dir = f"{dataset}_Single"
        
    all_targets = []

    for file in target_files:
        with open(f"NumPaths{data_dir}/{file}", "rb") as f:
            results = pickle.load(f)
            all_targets.append(list(results['networks']))

    return all_targets, data_dir

if __name__ == "__main__":
    mode = "batch"
    dataset = 3
    all_paths, data_dir = loadNumPaths(mode=mode, dataset=dataset)

    print(f'Dataset: NumPaths{data_dir}')
    for i in range(len(all_paths)):
        print('------------------------------')
        print(f"Target {i+1}: {len(all_paths[i])} pathways")
        sizes = [len(p) for p in all_paths[i]]
        print(f"  Size range: {min(sizes)}-{max(sizes)}, mean: {np.mean(sizes):.1f}")