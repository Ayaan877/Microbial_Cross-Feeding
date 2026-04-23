
import sys
import time
from load_data import *
from pathlib import Path
from datetime import datetime
from generate_crossfeeding_pairs import generate_crossfeeding_pairs

if __name__ == "__main__":
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Args supplied by PBS script (see run_crossNets.pbs)
    autonet_id    = sys.argv[1]        # autonet version
    crossnet_id   = sys.argv[2]        # output crossnet version label
    exchanged_met = sys.argv[3].lower()  # byp | int
    n_target      = int(sys.argv[4])
    n_workers     = int(sys.argv[5])

    if exchanged_met not in ("byp", "int"):
        raise ValueError(f"Invalid exchanged_met: '{exchanged_met}'. Must be 'byp' or 'int'.")

    output_dir = Path("data/networks")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"crossnets_rs_P_v{autonet_id}_{exchanged_met}_v{crossnet_id}.pkl"

    load_path = f"data/networks/autonets_rs_P_v{autonet_id}.pkl"
    with open(load_path, "rb") as f:
            all_autonets = pickle.load(f)

    print(f"Generating revScope crossfeeder networks from {load_path}...")
    print(f"  Output: {output_path}")

    start_time = time.time()

    if exchanged_met == "byp":
        pairs = generate_crossfeeding_pairs(
            all_autonets, rxnMat, prodMat, sumRxnVec,
            nutrientSet, Currency, Core,
            n_target=n_target,
            n_workers=n_workers,
            save_path=output_path,
            use_byproducts=True)
    else:
        pairs = generate_crossfeeding_pairs(
            all_autonets, rxnMat, prodMat, sumRxnVec,
            nutrientSet, Currency, Core,
            n_target=n_target,
            n_workers=n_workers,
            save_path=output_path,
            use_byproducts=False)

    total_time = time.time() - start_time

    print(f"Final: {len(pairs)} unique cross-feeding pairs generated")
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

