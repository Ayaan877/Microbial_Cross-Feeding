import pickle
import numpy as np

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