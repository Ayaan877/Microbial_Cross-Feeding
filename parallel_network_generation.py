import sys
import pickle
from generate_subgraphs import generate_pruned_networks
from load_data import *

with open("inv_met_map.pkl", "rb") as f:
    inv_met_map = pickle.load(f)

if __name__ == "__main__":

    if len(sys.argv) != 2:
        raise ValueError("Usage: python run_target.py <target_ID>")
    
    target_name = sys.argv[1]
    target = met_map[target_name]
    target_id = inv_met_map[target]

    print(f"Running target: {target}")
    print(f"Target ID: {target_id}")

    variants = generate_pruned_networks(target, rxnMat, prodMat, sumRxnVec, Energy, 
                                        Currency, n_variants=5, n_cores=5)

    with open(f"{target_id}_MinNets.pkl", "wb") as f:
        pickle.dump({"target": target, "target_id": target_id, "variants": variants}, f)

    print(f"Saved {len(variants)} variants to {target_id}_MinNets.pkl")
    print("Finished.")
