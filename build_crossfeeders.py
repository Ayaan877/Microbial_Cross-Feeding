
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
    exchanged_met = sys.argv[3]

    output_dir = Path("CrossNets_revScope")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"CrossNets_revScope{dataset_id}_{exchanged_met}{crossnet_id}.pkl"

    load_path = f"AutoNets_revScope/AutoNets_revScope_{dataset_id}.pkl"
    with open(load_path, "rb") as f:
            all_autonets = pickle.load(f)

    print(f"Generating revScope crossfeeder networks from {load_path}...")
    print(f"  Output: {output_path}")

    start_time = time.time()

    if exchanged_met.lower() == "byp":
        pairs = generate_crossfeeding_pairs(
            all_autonets, rxnMat, prodMat, sumRxnVec,
            nutrientSet, Currency, Core,
            n_target=50000,
            n_workers=32,
            save_path=output_path,
            use_byproducts=True)
    elif exchanged_met.lower() == "int":
        pairs = generate_crossfeeding_pairs(
            all_autonets, rxnMat, prodMat, sumRxnVec,
            nutrientSet, Currency, Core,
            n_target=50000,
            n_workers=32,
            save_path=output_path,
            use_byproducts=False)
    else:
        raise ValueError(f"Invalid exchanged_met: {exchanged_met}. Must be 'byp' or 'int'.")

    total_time = time.time() - start_time

    print(f"Final: {len(pairs)} unique cross-feeding pairs generated")
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

