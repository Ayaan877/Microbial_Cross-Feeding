import time
from pathlib import Path
from datetime import datetime
from load_data import *

# --- Configuration ---
MODE    = "NumPaths"  # "NumPaths" | "RevScope"

# --- NumPaths settings ---
BATCH_MODE              = "batch"     # "batch" | "single"
PRUNING                 = "noprune"   # "prune" | "noprune"
INPUT_DATASET_NUM       = "3"         # input dataset number
OUTPUT_DATASET_NUM      = "1"         # output dataset label
N_PROCESSES             = 32          # parallel worker processes

# --- RevScope settings ---
DATASET_ID   = "2"         # output filename label
N_TARGET     = 50000       # target number of unique networks
N_WORKERS    = 32          # parallel worker processes

# =============================================================================

if __name__ == "__main__":
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    start_time = time.time()

    output_dir = Path("data/networks")
    output_dir.mkdir(parents=True, exist_ok=True)

    if MODE == "RevScope":
        from generate_revScope_autoNets import generate_revScopeAutoNets

        output_path = output_dir / f"autonets_rs_P_v{DATASET_ID}.pkl"

        print(f"Generating revScope autonomous networks...")
        print(f"  Output: {output_path}")

        results = generate_revScopeAutoNets(
            rxnMat, prodMat, sumRxnVec, nutrientSet, Currency, Core,
            n_target=N_TARGET, n_workers=N_WORKERS,
            save_path=output_path)

        total_time = time.time() - start_time
        print(f"Saved {len(results)} unique autonets to {output_path}")

    else:
        if MODE == "NumPaths":
            from load_numpaths import loadNumPaths
            from generate_numPath_autoNets import allAutonomousNetworks
            all_paths = loadNumPaths(mode=BATCH_MODE, dataset=INPUT_DATASET_NUM)

        pruning = PRUNING.lower()
        prune_suffix = "P" if pruning == "prune" else "NP"
        output_file = f"autonets_np_{BATCH_MODE}_{prune_suffix}_pv{INPUT_DATASET_NUM}_v{OUTPUT_DATASET_NUM}.pkl"
        output_path = output_dir / output_file

        do_prune = pruning == "prune"
        print(f"Generating {'pruned' if do_prune else 'unpruned'} autonomous networks from paths (pruner={BATCH_MODE}, v{INPUT_DATASET_NUM})...")
        print(f"  Output: {output_path}")

        AutoNets = allAutonomousNetworks(all_paths, rxnMat, prodMat, sumRxnVec,
                                        nutrientSet, Currency, Core, prune=do_prune,
                                        n_processes=N_PROCESSES, save_path=output_path)

        total_time = time.time() - start_time
        if AutoNets:
            print(f"Saved {len(AutoNets)} autonomous networks to {output_file}")
        else:
            print(f"No autonomous networks generated")

    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
