import pickle
import numpy as np

def loadPaths(mode, dataset):
    if mode == "batch":
        data_dir = f"{dataset}_Batch"
        target_files = [
            "C00009_Batch_MinNets.pkl",
            "C00013_Batch_MinNets.pkl",
            "C00022_Batch_MinNets.pkl",
            "C00025_Batch_MinNets.pkl",
            "C00041_Batch_MinNets.pkl",
            "C00065_Batch_MinNets.pkl",
            "C00097_Batch_MinNets.pkl",
            "C00117_Batch_MinNets.pkl"]
        
    elif mode == "single":
        data_dir = f"{dataset}_Single"
        target_files = [
            "C00009_Single_MinNets.pkl",
            "C00013_Single_MinNets.pkl",
            "C00022_Single_MinNets.pkl",
            "C00025_Single_MinNets.pkl",
            "C00041_Single_MinNets.pkl",
            "C00065_Single_MinNets.pkl",
            "C00097_Single_MinNets.pkl",
            "C00117_Single_MinNets.pkl"]
        
    elif mode == "simple_single":
        data_dir = f"{dataset}_SimpleSingle"
        target_files = [
            "C00009_SimpleSingle_MinNets.pkl",
            "C00013_SimpleSingle_MinNets.pkl",
            "C00022_SimpleSingle_MinNets.pkl",
            "C00025_SimpleSingle_MinNets.pkl",
            "C00041_SimpleSingle_MinNets.pkl",
            "C00065_SimpleSingle_MinNets.pkl",
            "C00097_SimpleSingle_MinNets.pkl",
            "C00117_SimpleSingle_MinNets.pkl"]

    all_targets = []

    for file in target_files:
        with open(f"MinNets{data_dir}/{file}", "rb") as f:
            variants = pickle.load(f)
            all_targets.append(list(variants))

    return all_targets, data_dir

if __name__ == "__main__":
    mode = "batch"
    all_paths, data_dir = loadPaths(mode=mode, dataset=5)
    print(f'Dataset: MinNets{data_dir}')
    for i in range(len(all_paths)):
        print('------------------------------')
        print(f"Target {i+1}:")
        for j in range(len(all_paths[i])):
            print(len(all_paths[i][j]))