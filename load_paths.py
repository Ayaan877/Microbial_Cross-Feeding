import pickle
import numpy as np

def load_paths(mode):
    if mode == "batch":
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
        target_files = [
            "C00009_Single_MinNets.pkl",
            "C00013_Single_MinNets.pkl",
            "C00022_Single_MinNets.pkl",
            "C00025_Single_MinNets.pkl",
            "C00041_Single_MinNets.pkl",
            "C00065_Single_MinNets.pkl",
            "C00097_Single_MinNets.pkl",
            "C00117_Single_MinNets.pkl"]

    all_targets = []

    for file in target_files:
        with open(file, "rb") as f:
            variants = pickle.load(f)
            all_targets.append(variants)

    return all_targets