import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_runs(results_dir):
    """{experiment_name: {metric: (steps, values)}}"""
    runs = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "metrics_*.jsonl"))):
        name = os.path.basename(path)[len("metrics_") : -len(".jsonl")]
        series = defaultdict(lambda: ([], []))
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                step = row.get("step", 0)
                for key, val in row.items():
                    if key == "step" or not isinstance(val, (int, float)):
                        continue
                    series[key][0].append(step)
                    series[key][1].append(float(val))
        runs[name] = series
    return runs


def smooth(vals, k=5):
    if len(vals) < k:
        return vals
    out, s = [], 0.0
    q = []
    for v in vals:
        q.append(v)
        s += v
        if len(q) > k:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


def cmp_plot(runs, metric, out_path, title, ylabel, logy=False, smooth_k=0):
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, series in runs.items():
        if metric not in series:
            continue
        steps, vals = series[metric]
        if smooth_k:
            vals = smooth(vals, smooth_k)
        ax.plot(steps, vals, label=name, linewidth=1.2)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def final_summary(runs, out_path):
    """Финальные значения ключевых метрик каждого рана + таблица-картинка."""
    keys = [
        "val_loss", "val_perplexity", "word_fraction",
        "gen_tokens_per_sec_cached", "gen_tokens_per_sec_uncached",
        "gen_kv_cache_mb", "gpu_mem_mb", "tokens_per_sec",
        "moe_balance_loss", "moe_load_std", "n_params_total", "n_params_active",
    ]
    labels = {
        "val_loss": "val loss (best-final)", "val_perplexity": "val ppl",
        "word_fraction": "word fraction",
        "gen_tokens_per_sec_cached": "gen tok/s (KV-cache)",
        "gen_tokens_per_sec_uncached": "gen tok/s (no cache)",
        "gen_kv_cache_mb": "KV cache, MB",
        "gpu_mem_mb": "train peak mem, MB",
        "tokens_per_sec": "train tok/s",
        "moe_balance_loss": "MoE balance loss",
        "moe_load_std": "MoE load std",
        "n_params_total": "params, total M",
        "n_params_active": "params, active M",
    }
    rows = {}
    for name, series in runs.items():
        row = {}
        for k in keys:
            if k not in series:
                continue
            steps, vals = series[k]
            v = min(vals) if k == "val_loss" else vals[-1]
            row[k] = v
        rows[name] = row

    text_lines = []
    for name, row in rows.items():
        text_lines.append(name)
        for k in keys:
            if k in row:
                text_lines.append(f"  {labels[k]}: {row[k]:.4g}")
    with open(out_path.replace(".png", ".txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(text_lines))
    print(f"saved {out_path.replace('.png', '.txt')}")

    # png-таблица
    n_cols = len(keys) + 1
    n_rows = len(rows) + 1
    fig, ax = plt.subplots(figsize=(2.2 * n_cols, 0.6 * n_rows))
    ax.axis("off")
    cell_text = [["experiment"] + [labels[k] for k in keys]]
    for name in rows:
        r = ["-" if k not in rows[name] else f"{rows[name][k]:.4g}" for k in keys]
        cell_text.append([name] + r)
    table = ax.table(
        cellText=cell_text[1:],
        colLabels=cell_text[0],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/plots")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    runs = load_runs(args.results)
    if not runs:
        print("no metrics files found")
        return
    print("runs:", ", ".join(runs))

    cmp_plot(runs, "train_loss", os.path.join(args.out, "cmp_train_loss.png"),
             "Train loss", "loss", smooth_k=5)
    cmp_plot(runs, "val_loss", os.path.join(args.out, "cmp_val_loss.png"),
             "Validation loss", "loss")
    cmp_plot(runs, "val_perplexity", os.path.join(args.out, "cmp_val_ppl.png"),
             "Validation perplexity", "perplexity")
    cmp_plot(runs, "word_fraction", os.path.join(args.out, "cmp_word_fraction.png"),
             "Generation quality: dictionary word fraction", "fraction")
    cmp_plot(runs, "tokens_per_sec", os.path.join(args.out, "cmp_throughput.png"),
             "Training throughput", "tokens / sec", smooth_k=5)
    cmp_plot(runs, "gpu_util", os.path.join(args.out, "cmp_gpu_util.png"),
             "GPU utilization", "%", smooth_k=5)
    cmp_plot(runs, "moe_balance_loss", os.path.join(args.out, "cmp_moe_balance.png"),
             "MoE balance loss", "loss", logy=True)
    cmp_plot(runs, "moe_load_std", os.path.join(args.out, "cmp_moe_load_std.png"),
             "MoE load std (0 = perfect balance)", "std")
    final_summary(runs, os.path.join(args.out, "summary.png"))

    plt.close("all")


if __name__ == "__main__":
    main()
