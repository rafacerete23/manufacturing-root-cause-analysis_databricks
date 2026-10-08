"""Run the causal root-cause analysis core outside Databricks (no Spark, MLflow or Unity Catalog).

Reuses generate_data() from 99_utils.ipynb and the causal graph from
01_causal_graph.ipynb, then checks the two results the notebooks teach:
  1. anomaly attribution for one defective part (02/04)
  2. distribution-change attribution after a worker-mix shift (03) - the
     data generator changes only p_worker, so `worker` must come out on top.

Usage:  pip install -r local/requirements-local.txt
        python local/smoke_test.py          # exact tests, ~90 s
        python local/smoke_test.py --fast   # approximate tests, ~30 s (3x faster)

--fast swaps the kernel independence test that decides which mechanisms changed
(57 of the 70 s in distribution_change) for its random-Fourier-feature
approximation, and draws 1,000 instead of 3,000 samples for anomaly attribution.
The root cause comes out the same, but the approximate test also flags one or
two unchanged mechanisms as changed: fine for iterating, not for conclusions.
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from dowhy import gcm
from dowhy.gcm.independence_test import approx_kernel_based

warnings.filterwarnings("ignore")
gcm.config.disable_progress_bars()
ROOT = Path(__file__).resolve().parent.parent

# generate_data() lives in the first cell of 99_utils.ipynb; it only touches Spark when train=True.
ns: dict = {}
exec("".join(json.loads((ROOT / "99_utils.ipynb").read_text())["cells"][0]["source"]), ns)
generate_data = ns["generate_data"]

DEPENDENCIES = {  # same graph as 01_causal_graph.ipynb
    "position_alignment": ["worker", "machine"],
    "force_torque": ["raw_material", "machine", "material"],
    "temperature": ["chamber_temperature", "chamber_humidity", "chamber_pressure"],
    "dimensions": ["position_alignment", "force_torque"],
    "torque_checks": ["force_torque", "temperature"],
    "visual_inspection": ["temperature"],
    "quality": ["dimensions", "torque_checks", "visual_inspection"],
}
graph = nx.DiGraph([(p, c) for c, ps in DEPENDENCIES.items() for p in ps])
nodes = list(graph.nodes)


def top(attr: dict, k: int = 3) -> list[tuple[str, float]]:
    vals = {n: float(np.ravel(v)[0]) for n, v in attr.items()}
    return sorted(vals.items(), key=lambda kv: -abs(kv[1]))[:k]


parser = argparse.ArgumentParser(description="Causal RCA smoke test outside Databricks")
parser.add_argument("--fast", action="store_true", help="approximate tests (see module docstring)")
fast = parser.parse_args().fast
change_tests = {"independence_test": approx_kernel_based, "conditional_independence_test": approx_kernel_based} if fast else {}
anomaly_samples_n = 1000 if fast else 3000
start = time.time()

train = generate_data(None, None, 5000, train=False)[nodes]
print(f"train: {len(train)} parts, defect rate {train['quality'].mean():.1%}")

np.random.seed(1)
scm = gcm.StructuralCausalModel(graph)
gcm.auto.assign_causal_mechanisms(scm, train)
gcm.fit(scm, train)

defect = train[train["quality"] == 1].head(1)
print("anomaly attribution, first defective part:", top(gcm.attribute_anomalies(scm, "quality", anomaly_samples=defect, num_distribution_samples=anomaly_samples_n)))

test = generate_data(None, None, 1000, p_worker=0.25, train=False)[nodes]
print(f"test (p_worker 0.75 -> 0.25): defect rate {test['quality'].mean():.1%}")
change = gcm.distribution_change(scm, train, test, target_node="quality",
                                 difference_estimation_func=lambda x, y: np.mean(y) - np.mean(x), **change_tests)
ranked = top(change, k=len(change))
print("distribution-change attribution:", ranked[:3])
assert ranked[0][0] == "worker", f"expected `worker` as the root cause of the shift, got {ranked[0][0]}"
print(f"OK: the shift is attributed to `worker`, the only mechanism the generator changed ({'fast' if fast else 'exact'} mode, {time.time() - start:.0f} s)")
