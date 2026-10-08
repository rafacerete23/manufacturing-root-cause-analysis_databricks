"""Exact vs --fast distribution-change attribution: speed against false positives.

dowhy's distribution_change first tests every node's causal mechanism for a
change (kernel independence tests), then attributes the shift in `quality`
only to the mechanisms that changed. The data generator changes exactly one
mechanism (`worker`, p 0.75 -> 0.25), so every other node it flags as changed
is a false positive. This runs both modes over several seeds and reports
runtime, whether `worker` ranks first, and which mechanisms each mode's
change test flagged (dowhy's `return_additional_info`).

Usage:  python local/compare_modes.py --seeds 3      (~6 minutes)
"""
import argparse
import time
import warnings

import numpy as np
import pandas as pd
from dowhy import gcm
from dowhy.gcm.independence_test import approx_kernel_based

import smoke_test as base  # graph, nodes and generate_data, shared with the smoke test

warnings.filterwarnings("ignore")
gcm.config.disable_progress_bars()

MODES = {
    "exact": {},
    "fast": {"independence_test": approx_kernel_based, "conditional_independence_test": approx_kernel_based},
}


def run(seed: int, mode: str) -> dict:
    np.random.seed(seed)
    train = base.generate_data(None, None, 5000, train=False)[base.nodes]
    test = base.generate_data(None, None, 1000, p_worker=0.25, train=False)[base.nodes]
    scm = gcm.StructuralCausalModel(base.graph)
    gcm.auto.assign_causal_mechanisms(scm, train)
    t0 = time.time()
    change, changed, _, _ = gcm.distribution_change(scm, train, test, target_node="quality",
                                                    difference_estimation_func=lambda x, y: np.mean(y) - np.mean(x),
                                                    return_additional_info=True, **MODES[mode])
    secs = time.time() - t0
    vals = {n: float(np.ravel(v)[0]) for n, v in change.items()}
    ranked = sorted(vals, key=lambda n: -abs(vals[n]))
    flagged = [n for n, c in changed.items() if c and n != "worker"]
    return {"seed": seed, "mode": mode, "seconds": round(secs, 1), "worker_first": ranked[0] == "worker",
            "worker_share": round(abs(vals["worker"]) / sum(abs(v) for v in vals.values()), 2),
            "worker_flagged": bool(changed["worker"]), "false_positives": len(flagged),
            "flagged": ", ".join(sorted(flagged)) or "-"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    n = ap.parse_args().seeds
    rows = [run(s, m) for s in range(n) for m in MODES]
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    print("\n", df.groupby("mode").agg(seconds=("seconds", "mean"), worker_first=("worker_first", "mean"),
                                       worker_share=("worker_share", "mean"), worker_flagged=("worker_flagged", "mean"),
                                       false_positives=("false_positives", "mean")).round(2))
