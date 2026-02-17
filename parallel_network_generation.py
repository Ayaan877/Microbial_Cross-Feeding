import sys
import pickle
import time
from datetime import datetime
from generate_subgraphs import generate_pruned_networks
from load_data import *

with open("inv_met_map.pkl", "rb") as f:
    inv_met_map = pickle.load(f)

if __name__ == "__main__":
    start_time = time.time()
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    target_name = sys.argv[1]
    target = met_map[target_name]
    target_id = inv_met_map[target]

    print(f"Running target: {target}")
    print(f"Target ID: {target_id}")

    variants = generate_pruned_networks(target, rxnMat, prodMat, sumRxnVec, Energy, 
                                        Currency, n_variants=5, n_cores=10)

    if variants:
        with open(f"{target_id}_MinNets.pkl", "wb") as f:
            pickle.dump(variants, f)

        print(f"Saved {len(variants)} variants to {target_id}_MinNets.pkl")
    else:
        print(f"No variants generated for {target_id}")

    total_time = time.time() - start_time
    print(f"Total time: {total_time/60:.2f} minutes")
    print("Finished.")
