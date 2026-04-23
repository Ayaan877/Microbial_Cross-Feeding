import sys
import time
from pathlib import Path
from datetime import datetime
from load_data import *

if __name__ == "__main__":
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Args supplied by PBS script (see run_autoNets.pbs)
    MODE = sys.argv[1]   # RevScope | NumPaths

    if MODE == "RevScope":
        # python build_autonomous_networks.py RevScope <dataset_id> <n_target> <n_workers>
        DATASET_ID = sys.argv[2]
        N_TARGET   = int(sys.argv[3])
        N_WORKERS  = int(sys.argv[4])
    else:
        # python build_autonomous_networks.py NumPaths <batch_mode> <pruning> <input_dataset> <output_dataset> <n_target> <n_workers>
        BATCH_MODE        = sys.argv[2]   # batch | single
        PRUNING           = sys.argv[3]   # prune | noprune
        INPUT_DATASET_ID  = sys.argv[4]
        OUTPUT_DATASET_ID = sys.argv[5]
        N_TARGET          = int(sys.argv[6])
        N_WORKERS         = int(sys.argv[7])

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
            from load_minPaths import loadNumPaths
            from generate_minPath_autoNets import generate_minPathAutoNets
            all_paths = loadNumPaths(mode=BATCH_MODE, dataset=INPUT_DATASET_ID)

        pruning = PRUNING.lower()
        prune_suffix = "P" if pruning == "prune" else "NP"
        output_file = f"autonets_np_{BATCH_MODE}_{prune_suffix}_pv{INPUT_DATASET_ID}_v{OUTPUT_DATASET_ID}.pkl"
        output_path = output_dir / output_file

        do_prune = pruning == "prune"
        print(f"Generating {'pruned' if do_prune else 'unpruned'} autonomous networks from paths (pruner={BATCH_MODE}, v{INPUT_DATASET_ID})...")
        print(f"  Output: {output_path}")

        AutoNets = generate_minPathAutoNets(all_paths, rxnMat, prodMat, sumRxnVec,
                                        nutrientSet, Currency, Core, prune=do_prune,
                                        n_target=N_TARGET, n_workers=N_WORKERS,
                                        save_path=output_path)

        total_time = time.time() - start_time
        if AutoNets:
            print(f"Saved {len(AutoNets)} autonomous networks to {output_file}")
        else:
            print(f"No autonomous networks generated")

    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
