def resolve_paths(d):
    typ, source, autonet_id, crossnet_id, met, pruner, pruning, pv, yield_mode, label = d[:10]
    subdir = d[10] if len(d) > 10 else ""
    suffix = "P" if pruning == "prune" else ("NP" if pruning == "noprune" else None)
    yields_base = f"data/yields/{subdir}" if subdir else "data/yields"
    if source == "rs":
        autoID = f"rs_P_v{autonet_id}"
    else:
        autoID = f"mp_{pruner}_{suffix}_pv{pv}_v{autonet_id}"
    if typ == "auto":
        if source == "rs":
            net_path = f"data/networks/autonets_rs_P_v{autonet_id}.pkl"
        else:
            net_path = f"data/networks/autonets_mp_{pruner}_{suffix}_pv{pv}_v{autonet_id}.pkl"
        if yield_mode == "stoich":
            yield_path = f"{yields_base}/stoich_auto_{autoID}.pkl"
        else:
            yield_path = f"{yields_base}/yields_auto_{autoID}_{yield_mode}.pkl"
    else:
        if source == "rs":
            net_path = f"data/networks/crossnets_rs_P_v{autonet_id}_{met}_v{crossnet_id}.pkl"
        else:
            net_path = f"data/networks/crossnets_mp_{pruner}_{suffix}_pv{pv}_v{autonet_id}_{met}_v{crossnet_id}.pkl"
        if yield_mode == "stoich":
            yield_path = f"{yields_base}/stoich_cross_{autoID}_{met}_v{crossnet_id}.pkl"
        else:
            yield_path = f"{yields_base}/yields_cross_{autoID}_{met}_v{crossnet_id}_{yield_mode}.pkl"
    return net_path, yield_path