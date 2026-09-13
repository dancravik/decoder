"""Smoke-тест: все конфиги на КРОШЕЧНОЙ модели (d_model=64, 2 слоя, 5 шагов,
batch 4) - только проверка корректности кода, без серьёзной нагрузки на CPU."""
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

import yaml

os.environ["CUDA_VISIBLE_DEVICES"] = ""

BASE = "configs/baseline_mha.yaml"
TINY = {
    "d_model": 64,
    "n_layers": 2,
    "n_heads": 4,
    "n_kv_heads": 2,
    "d_ff": 128,
    "seq_len": 64,
    "moe": {"d_ff_shared": 64, "d_ff": 32},
}
TINY_TRAIN = {
    "max_steps": 5,
    "batch_size": 4,
    "warmup_steps": 1,
    "eval_interval": 2,
    "sample_interval": 4,
    "ckpt_interval": 100,
    "torch_compile": False,
    "sample_length": 16,
}


def merge(a, b):
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out


def main():
    results_dir = tempfile.mkdtemp(prefix="smoke_")
    ok, failed = [], []
    for cfg_path in sorted(glob.glob("configs/*.yaml")):
        name = os.path.splitext(os.path.basename(cfg_path))[0]
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        with open(BASE) as f:
            base = yaml.safe_load(f)
        if cfg.pop("base", None):
            cfg = merge(base, cfg)
        cfg["experiment_name"] = name
        structural = dict(cfg["structural"])
        # аккуратно уменьшаем: сохраняем типы attention/ffn/moe/balance
        structural.update(TINY)
        if cfg["structural"].get("attention") == "mqa":
            structural.pop("n_kv_heads", None)  # mqa игнорирует n_kv_heads
        cfg["structural"] = structural
        cfg["training"] = {**cfg.get("training", {}), **TINY_TRAIN}
        cfg.pop("comet", None)  # без comet в тесте
        tmp_cfg = os.path.join(results_dir, f"cfg_{name}.yaml")
        with open(tmp_cfg, "w") as f:
            yaml.safe_dump(cfg, f)

        print(f"=== smoke: {name} ===", flush=True)
        env = {**os.environ, "CONFIG": tmp_cfg, "RESULTS_DIR": results_dir}
        r = subprocess.run(
            [sys.executable, "train.py"],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if r.returncode != 0:
            print(r.stdout[-2000:])
            print(r.stderr[-2000:])
            print(f"FAIL: {name}")
            failed.append(name)
            continue

        with open(os.path.join(results_dir, f"metrics_{name}.jsonl")) as f:
            rows = [json.loads(l) for l in f]
        keys = sorted({k for row in rows for k in row})
        has_val = any("val_loss" in row for row in rows)
        has_gen = any("gen_tokens_per_sec_cached" in row for row in rows)
        print(f"OK: {name} | metrics: {keys} | val: {has_val} | gen-bench: {has_gen}")
        if not has_val or not has_gen:
            failed.append(name)
        else:
            ok.append(name)

    print(f"\nPASSED: {ok}")
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    print(f"ALL {len(ok)} CONFIGS PASSED")
    shutil.rmtree(results_dir)


if __name__ == "__main__":
    main()
