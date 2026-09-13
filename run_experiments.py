import argparse
import glob
import os
import subprocess
import sys

CONFIGS = [
    "baseline_mha",
    "mha_flash",
    "gqa",
    "gqa_flash",
    "mqa",
    "moe_aux_free",
    "moe_aux_loss",
    "modern_all",
]


def run_all(selected):
    for name in selected:
        path = os.path.join("configs", f"{name}.yaml")
        print(f"\n{'='*70}\n=== running {name} ===\n{'='*70}", flush=True)
        env = {**os.environ, "CONFIG": path, "RESULTS_DIR": "results"}
        res = subprocess.run([sys.executable, "train.py"], env=env)
        if res.returncode != 0:
            print(f"experiment {name} FAILED with code {res.returncode}")
            sys.exit(res.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiments",
        nargs="*",
        help=f"names from configs/ ({', '.join(CONFIGS)}); default = all",
    )
    args = parser.parse_args()
    selected = [e for e in args.experiments if e]
    if not selected:
        existing = sorted(
            os.path.splitext(os.path.basename(p))[0] for p in glob.glob("configs/*.yaml")
        )
        selected = existing
    run_all(selected)
    print("\nall experiments done")
    subprocess.run([sys.executable, "plot_metrics.py"], check=False)


if __name__ == "__main__":
    main()
