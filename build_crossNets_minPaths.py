import sys
import time
import pickle
from load_data import *
from pathlib import Path
from datetime import datetime
from load_minPaths import loadMinPaths
from generate_crossNets_minPaths import generate_crossNets_minPaths

if __name__ == "__main__":
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Usage: build_crossNets_minPaths.py <pruner> <paths_version> <autonet_id> <crossnet_id> <exchanged_met> <n_target> <n_workers>
    #
    # pruner        : batch | single
    # paths_version : minpath dataset version (e.g. 1)
    # autonet_id    : autonet version label (for naming consistency)
    # crossnet_id   : output crossnet version label
    # exchanged_met : byp | int
    # n_target      : number of unique cross-feeding pairs to generate
    # n_workers     : number of parallel workers
    #
    # Output file: crossnets_mp_<pruner>_NP_pv<paths_version>_v<autonet_id>_<exchanged_met>_v<crossnet_id>.pkl
    # "NP" (noprune) identifies these networks as built directly from raw MinPaths
    # without any post-assembly pruning.

    pruner        = sys.argv[1]          # batch | single
    paths_version = sys.argv[2]          # minpath dataset version
    autonet_id    = sys.argv[3]          # autonet version label
    crossnet_id   = sys.argv[4]          # output crossnet version label
    exchanged_met = sys.argv[5].lower()  # byp | int
    n_target      = int(sys.argv[6])
    n_workers     = int(sys.argv[7])

    if pruner not in ("batch", "single"):
        raise ValueError(f"Invalid pruner: '{pruner}'. Must be 'batch' or 'single'.")
    if exchanged_met not in ("byp", "int"):
        raise ValueError(f"Invalid exchanged_met: '{exchanged_met}'. Must be 'byp' or 'int'.")

    output_dir = Path("data/networks")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / (
        f"crossnets_mp_{pruner}_NP_pv{paths_version}_v{autonet_id}_{exchanged_met}_v{crossnet_id}.pkl"
    )

    print(f"Loading MinPaths (pruner={pruner}, version={paths_version})...")
    all_paths = loadMinPaths(mode=pruner, dataset=paths_version)
    print(f"  Loaded {len(all_paths)} core targets, "
          f"{sum(len(p) for p in all_paths)} total pathways.")
    print(f"Output: {output_path}")

    use_byproducts = (exchanged_met == "byp")

    start_time = time.time()

    pairs = generate_crossNets_minPaths(
        all_paths, rxnMat, prodMat, sumRxnVec,
        nutrientSet, Currency, Core,
        n_target=n_target,
        n_workers=n_workers,
        save_path=output_path,
        use_byproducts=use_byproducts)

    total_time = time.time() - start_time

    print(f"Final: {len(pairs)} unique cross-feeding pairs generated")
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
