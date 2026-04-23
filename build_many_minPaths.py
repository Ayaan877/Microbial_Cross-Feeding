import sys
import time
from pathlib import Path
from datetime import datetime
from generate_many_minPaths import generate_pruned_networks
from load_data import *

if __name__ == "__main__":
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Args supplied by PBS script (see run_many_minPaths.pbs)
    target_name       = sys.argv[1]        # KEGG metabolite ID
    mode              = sys.argv[2].lower()  # batch | single
    path_id           = sys.argv[3]        # output version label
    N_WORKERS           = int(sys.argv[4])
    max_attempts      = int(sys.argv[5])
    plateau_window    = int(sys.argv[6])
    plateau_threshold = int(sys.argv[7])

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

    output_file = f"paths_{mode}_{target_id}_v{path_id}.pkl"
    output_dir = Path("data/paths")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_file

    start_time = time.time()
    results = generate_pruned_networks(target, rxnMat, prodMat, sumRxnVec, nutrientSet,
                                       Currency, n_workers=N_WORKERS, randMinNetwork=randMinNetwork,
                                       save_path=output_path, max_attempts=max_attempts,
                                       plateau_window=plateau_window,
                                       plateau_threshold=plateau_threshold)

    if results:
        print(f"{len(results['networks'])} variants generated in {results['attempts'][-1]} attempts")

    total_time = time.time() - start_time
    print(f"Total time: {total_time/60:.2f} minutes")