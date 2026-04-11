import sys
import time
from pathlib import Path
from datetime import datetime
from load_data import *
from generate_revScope_autoNets import generate_minimal_autonets

if __name__ == "__main__":
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    dataset_id = sys.argv[1]

    output_dir = Path("AutoNets_revScope")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"AutoNets_revScope_{dataset_id}.pkl"

    print(f"Generating revScope autonomous networks...")
    print(f"  Output: {output_path}")

    start_time = time.time()

    results = generate_minimal_autonets(
        rxnMat, prodMat, sumRxnVec, nutrientSet, Currency, Core,
        n_target=50000, n_workers=32,
        save_path=output_path)

    total_time = time.time() - start_time

    print(f"Saved {len(results['networks'])} unique autonets to {output_path}")
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
