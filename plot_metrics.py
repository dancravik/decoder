import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_metrics(path):
    steps = {"train_loss": [], "val_loss": [], "val_perplexity": [], "word_fraction": []}
    vals = {"train_loss": [], "val_loss": [], "val_perplexity": [], "word_fraction": []}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            step = row.get("step", 0)
            for key in steps:
                if key in row:
                    steps[key].append(step)
                    vals[key].append(row[key])
    return steps, vals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="results/metrics.jsonl")
    parser.add_argument("--out", default="results/plots")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    steps, vals = load_metrics(args.metrics)

    if steps["train_loss"]:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(steps["train_loss"], vals["train_loss"], label="train loss", linewidth=0.8)
        ax.plot(steps["val_loss"], vals["val_loss"], label="val loss", marker="o")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title("Train / Val loss")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "loss.png"), dpi=150)
        print(f"saved {os.path.join(args.out, 'loss.png')}")

    if steps["val_perplexity"]:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(steps["val_perplexity"], vals["val_perplexity"], label="val perplexity", marker="o")
        ax.set_xlabel("step")
        ax.set_ylabel("perplexity")
        ax.set_title("Validation perplexity")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "perplexity.png"), dpi=150)
        print(f"saved {os.path.join(args.out, 'perplexity.png')}")

    if steps["word_fraction"]:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(steps["word_fraction"], vals["word_fraction"], marker="o", color="tab:green")
        ax.set_xlabel("step")
        ax.set_ylabel("fraction of real words")
        ax.set_title("Generation quality: share of dictionary words in samples")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "word_fraction.png"), dpi=150)
        print(f"saved {os.path.join(args.out, 'word_fraction.png')}")

    plt.close("all")


if __name__ == "__main__":
    main()
