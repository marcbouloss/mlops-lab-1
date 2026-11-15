"""Lab 5: quality gate before moving the `champion` alias.

    uv run python src/food11/promote.py --version 8

The candidate becomes champion only if it beats the current champion by more than
the noise of the test set: acc_candidate >= acc_champion + 2 * se, with
se = sqrt(p * (1 - p) / n), p = acc_champion, n = size of the test split.
Otherwise it gets the `challenger` alias. The decision is written as tags on the
candidate version in both cases.
"""
from __future__ import annotations

import argparse
import math
import os

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

NAME = "food11"
METRIC = "test_accuracy"
N_TEST = 1096  # data/food11_processed_mini/evaluation after de-duplication (Lab 5)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", required=True, help="candidate model version of food11")
    ap.add_argument("--n-test", type=int, default=N_TEST, help="number of images in the test split")
    args = ap.parse_args()

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    client = MlflowClient()

    cand = client.get_model_version(NAME, args.version)
    acc_cand = client.get_run(cand.run_id).data.metrics[METRIC]
    try:
        champ = client.get_model_version_by_alias(NAME, "champion")
    except MlflowException:
        champ = None  # empty registry: nothing to beat

    if champ is None:
        acc_champ, se, threshold, promote = float("nan"), float("nan"), float("nan"), True
    else:
        acc_champ = client.get_run(champ.run_id).data.metrics[METRIC]
        se = math.sqrt(acc_champ * (1 - acc_champ) / args.n_test)
        threshold = acc_champ + 2 * se
        promote = acc_cand >= threshold

    decision = "promoted" if promote else "challenger"
    client.set_registered_model_alias(NAME, "champion" if promote else "challenger", cand.version)
    for key, value in {
        "gate_decision": decision,
        "gate_acc_candidate": f"{acc_cand:.4f}",
        "gate_acc_champion": f"{acc_champ:.4f}",
        "gate_champion_version": champ.version if champ else "none",
        "gate_threshold": f"{threshold:.4f}",
        "gate_n_test": str(args.n_test),
    }.items():
        client.set_model_version_tag(NAME, cand.version, key, value)

    print(f"candidate v{cand.version}: {METRIC}={acc_cand:.4f}")
    if champ:
        print(f"champion  v{champ.version}: {METRIC}={acc_champ:.4f}  se={se:.4f}  "
              f"threshold=acc_champion+2se={threshold:.4f}")
    print(f"decision: {decision} -> alias '{'champion' if promote else 'challenger'}' set on v{cand.version}")


if __name__ == "__main__":
    main()
