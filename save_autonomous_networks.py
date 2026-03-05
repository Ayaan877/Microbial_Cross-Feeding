import sys
import pickle
import time
from pathlib import Path
from datetime import datetime
from load_paths import loadPaths
from generate_networks import allAutonomousNetworks
from load_data import *

if __name__ == "__main__":
    start_time = time.time()
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    mode = sys.argv[1]
    all_paths, data_dir = loadPaths(mode=mode, dataset=6)

    AutoNets = allAutonomousNetworks(all_paths, rxnMat, prodMat, sumRxnVec, 
                                     nutrientSet, Currency, Core)
    output_file = f"AutoNets{data_dir}.pkl"
    output_dir = Path(f"AutoNets{data_dir}")
    output_dir.mkdir(exist_ok=True)
    
    if AutoNets:
        output_path = output_dir / output_file
        with open(output_path, "wb") as f:
            pickle.dump(AutoNets, f)

        print(f"Saved {len(AutoNets)} autonomous networks to {output_file}")
    else:
        print(f"No autonomous networks generated for MinNets{data_dir}")

    total_time = time.time() - start_time
    print(f"Total time: {total_time/60:.2f} minutes")
