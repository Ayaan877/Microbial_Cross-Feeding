import sys
import time
from pathlib import Path
from datetime import datetime
from load_paths import loadPaths
from generate_autoNets import allAutonomousNetworks
from load_data import *

if __name__ == "__main__":
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    mode = sys.argv[1]
    pruning = sys.argv[2].lower()
    dataset = sys.argv[3]
    all_paths, data_dir = loadPaths(mode=mode, dataset=dataset)

    start_time = time.time()
    if pruning == "prune":
        output_file = f"AutoNets{data_dir}.pkl"
        output_dir = Path(f"AutoNets{data_dir}_P")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / output_file

        print(f"Generating pruned autonomous networks from MinNets{data_dir}...")

        AutoNets = allAutonomousNetworks(all_paths, rxnMat, prodMat, sumRxnVec, 
                                        nutrientSet, Currency, Core, prune=True, 
                                        n_processes=32, save_path=output_path)
    if pruning == "noprune":
        output_file = f"AutoNets{data_dir}.pkl"
        output_dir = Path(f"AutoNets{data_dir}_NP")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / output_file

        print(f"Generating unpruned autonomous networks from MinNets{data_dir}...")

        AutoNets = allAutonomousNetworks(all_paths, rxnMat, prodMat, sumRxnVec, 
                                        nutrientSet, Currency, Core, prune=False, 
                                        n_processes=32, save_path=output_path)

    if AutoNets:
        print(f"Saved {len(AutoNets)} autonomous networks to {output_file}")
    else:
        print(f"No autonomous networks generated for MinNets{data_dir}")

    total_time = time.time() - start_time
    print(f"Total time: {total_time/60:.2f} minutes")
