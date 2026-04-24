import sys
import time
from load_data import *
from pathlib import Path
from datetime import datetime
from generate_crossNets import generate_crossNets

if __name__ == "__main__":
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Args supplied by PBS script (see run_crossfeeding_networks.pbs)
    # Usage (rs): build_crossNets.py rs  <autonet_id> <crossnet_id> <exchanged_met> <n_target> <n_workers>
    # Usage (mp): build_crossNets.py mp  <autonet_id> <crossnet_id> <exchanged_met> <n_target> <n_workers> <pruner> <pruning> <paths_version>
    source        = sys.argv[1]          # rs | mp
    autonet_id    = sys.argv[2]          # autonet version
    crossnet_id   = sys.argv[3]          # output crossnet version label
    exchanged_met = sys.argv[4].lower()  # byp | int
    n_target      = int(sys.argv[5])
    n_workers     = int(sys.argv[6])

    if source not in ("rs", "mp"):
        raise ValueError(f"Invalid source: '{source}'. Must be 'rs' or 'mp'.")
    if exchanged_met not in ("byp", "int"):
        raise ValueError(f"Invalid exchanged_met: '{exchanged_met}'. Must be 'byp' or 'int'.")

    output_dir = Path("data/networks")
    output_dir.mkdir(parents=True, exist_ok=True)

    if source == "rs":
        load_path   = f"data/networks/autonets_rs_P_v{autonet_id}.pkl"
        output_path = output_dir / f"crossnets_rs_P_v{autonet_id}_{exchanged_met}_v{crossnet_id}.pkl"
    else:
        pruner        = sys.argv[7]   # batch | single
        pruning       = sys.argv[8]   # prune | noprune
        paths_version = sys.argv[9]   # minpath dataset version
        prune_suffix  = "P" if pruning == "prune" else "NP"
        load_path   = f"data/networks/autonets_mp_{pruner}_{prune_suffix}_pv{paths_version}_v{autonet_id}.pkl"
        output_path = output_dir / f"crossnets_mp_{pruner}_{prune_suffix}_pv{paths_version}_v{autonet_id}_{exchanged_met}_v{crossnet_id}.pkl"
    with open(load_path, "rb") as f:
        all_autonets = pickle.load(f)

    print(f"Generating crossfeeder networks from {load_path}...")
    print(f"  Output: {output_path}")

    start_time = time.time()

    if exchanged_met == "byp":
        pairs = generate_crossNets(
            all_autonets, rxnMat, prodMat, sumRxnVec,
            nutrientSet, Currency, Core,
            n_target=n_target,
            n_workers=n_workers,
            save_path=output_path,
            use_byproducts=True)
    else:
        pairs = generate_crossNets(
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

