
import sys
import time
from load_data import *
from pathlib import Path
from datetime import datetime
from generate_crossfeeding_pairs import generate_crossfeeding_pairs


if __name__ == "__main__":
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    dataset_id = sys.argv[1]
    crossnet_id = sys.argv[2]

    output_dir = Path("CrossNets_revScope")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"CrossNets_revScope{dataset_id}_{crossnet_id}.pkl"

    with open(f"AutoNets_revScope/AutoNets_revScope_{dataset_id}.pkl", "rb") as f:
            all_autonets = pickle.load(f)

    print(f"Generating revScope crossfeeder networks...")
    print(f"  Output: {output_path}")

    start_time = time.time()

    pairs = generate_crossfeeding_pairs(
        all_autonets, rxnMat, prodMat, sumRxnVec,
        nutrientSet, Currency, Core,
        n_target=50000,
        n_workers=32,
        save_path=output_path)

    total_time = time.time() - start_time

    print(f"Final: {len(pairs)} unique cross-feeding pairs generated")
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

