import sys
import pickle
import time
from pathlib import Path
from datetime import datetime
from load_data import *

if __name__ == "__main__":
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    target_name = sys.argv[1]
    mode = sys.argv[2].lower()
    dataset = sys.argv[3]

    target = met_map[target_name]
    target_id = inv_met_map[target]

    print(f"Running target: {target}")
    print(f"Target ID: {target_id}")

    if mode == "batch":
        from batch_pruning import randMinNetwork
        from generate_minPaths import generate_pruned_networks
        output_file = f"{target_id}_Batch_MinNets.pkl"
        output_dir = Path(f"MinNets{dataset}_Batch")
        output_dir.mkdir(exist_ok=True)

    elif mode == "simple_single":
        from single_pruning import randMinNetwork
        from generate_minPaths import generate_pruned_networks
        output_file = f"{target_id}_Single_MinNets.pkl"
        output_dir = Path(f"MinNets{dataset}_Single")
        output_dir.mkdir(exist_ok=True)

    else:
        raise ValueError("Mode must be 'batch' or 'single'")
    
    start_time = time.time()
    variants = generate_pruned_networks(target, rxnMat, prodMat, sumRxnVec, nutrientSet, Currency,
                                        n_variants=4, n_cores=4, randMinNetwork=randMinNetwork)

    if variants:
        output_path = output_dir/output_file
        with open(output_path, "wb") as f:
            pickle.dump(variants, f)

        print(f"Saved {len(variants)} variants to {output_file}")
    else:
        print(f"No variants generated for {target_id}")

    total_time = time.time() - start_time
    print(f"Total time: {total_time/60:.2f} minutes")
