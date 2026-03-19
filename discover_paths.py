import sys
import time
from pathlib import Path
from datetime import datetime
from path_discovery_rate import generate_pruned_networks
from load_data import *

if __name__ == "__main__":
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    target_name = sys.argv[1]
    mode = sys.argv[2].lower()
    target = met_map[target_name]
    target_id = inv_met_map[target]

    if mode == "batch":
        from batch_pruning import randMinNetwork
    elif mode == "single":
        from single_pruning import randMinNetwork
    else: 
        raise ValueError("Mode must be 'batch' or 'single'")
    
    print(f"Running target: {target}")
    print(f"Target ID: {target_id}")

    output_file = f"{target_id}_Pathways.pkl"
    output_dir = Path(f"NumPaths")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / output_file
    
    start_time = time.time()
    results = generate_pruned_networks(target, rxnMat, prodMat, sumRxnVec, nutrientSet, 
                                       Currency, n_cores=8, randMinNetwork=randMinNetwork,
                                       save_path=output_path)

    if results:
        print(f"{len(results['networks'])} variants generated in {results['attempts'][-1]} attempts")

    total_time = time.time() - start_time
    print(f"Total time: {total_time/60:.2f} minutes")