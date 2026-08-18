import json
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from decoder import (
    CharTokenizer,
    ShakespeareDataset,
    build_decoder_only,
    read_csv_text,
)

COMMON_WORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
    "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
    "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
    "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
    "is", "are", "was", "were", "am", "been", "being", "has", "had", "did",
    "does", "am", "here", "why", "didst", "thou", "thee", "thy", "hath", "dost",
    "shall", "art", "hast", "ere", "nay", "ay", "prithee", "sir", "madam", "lord",
    "lady", "king", "queen", "master", "mistress", "sweet", "love", "heart", "death",
    "life", "world", "man", "men", "woman", "women", "night", "day", "hand",
    "eye", "eyes", "face", "voice", "word", "words", "blood", "soul", "spirit",
    "god", "heaven", "hell", "fate", "fortune", "truth", "honor", "peace", "war",
    "fear", "hope", "joy", "sorrow", "grief", "pain", "tears", "smile", "kiss",
    "fire", "water", "earth", "air", "sun", "moon", "star", "stars", "light",
    "dark", "darkness", "shadow", "dream", "dreams", "sleep", "mind", "thought",
    "house", "home", "door", "room", "street", "city", "land", "country", "sea",
    "sword", "shield", "crown", "throne", "court", "duke", "earl", "knight", "soldier",
    "boy", "girl", "child", "children", "father", "mother", "son", "daughter", "brother",
    "sister", "friend", "friends", "enemy", "foe", "stranger", "guest", "servant", "fool",
    "fools", "knave", "witch", "ghost", "devil", "angel", "priest", "doctor", "nurse",
    "captain", "general", "messenger", "citizen", "citizens", "gentleman", "gentlemen",
    "come", "go", "stay", "speak", "hear", "tell", "say", "ask", "answer",
    "know", "think", "believe", "remember", "forget", "understand", "mean", "swear",
    "promise", "beg", "pray", "weep", "cry", "laugh", "sing", "dance", "fight",
    "kill", "die", "live", "run", "walk", "stand", "sit", "fall", "rise",
    "bear", "hold", "keep", "give", "take", "bring", "send", "show", "find",
    "seek", "lose", "win", "pay", "buy", "sell", "break", "mend", "build",
    "burn", "eat", "drink", "rest", "watch", "wait", "haste", "follow", "lead",
    "serve", "rule", "reign", "command", "obey", "defy", "betray", "forgive", "pardon",
    "bless", "curse", "praise", "blame", "thank", "welcome", "farewell", "adieu",
    "hello", "goodbye", "yes", "no", "never", "ever", "always", "soon", "anon",
    "tomorrow", "tonight", "today", "yesterday", "morning", "evening", "hour", "hours",
    "minute", "moment", "while", "long", "short", "great", "small", "little",
    "big", "high", "low", "old", "young", "fair", "foul", "true", "false",
    "wise", "mad", "brave", "bold", "gentle", "kind", "cruel", "proud", "base",
    "noble", "rich", "poor", "happy", "sad", "glad", "merry", "weary", "sick",
    "whole", "strong", "weak", "quick", "slow", "hot", "cold", "warm", "cool",
    "soft", "hard", "rough", "smooth", "bright", "pale", "red", "white", "black",
    "green", "blue", "gold", "silver", "iron", "stone", "wood", "flesh", "bone",
    "wine", "bread", "meat", "milk", "salt", "sugar", "honey", "rose", "lily",
    "flower", "flowers", "tree", "trees", "garden", "field", "hill", "valley", "river",
    "road", "path", "gate", "wall", "tower", "castle", "palace", "church", "prison",
}

try:
    from comet_ml import Experiment, API

    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_lr(step, base_lr, warmup_steps, max_steps):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return base_lr * 0.1
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return base_lr * 0.1 + 0.9 * base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def word_fraction(tokenizer, generated_ids):
    text = tokenizer.decode(generated_ids)
    words = [w for w in "".join(c if c.isalpha() else " " for c in text.lower()).split()]
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in COMMON_WORDS)
    return hits / len(words)


@torch.no_grad()
def estimate_loss(model, dataset, device, batch_size, ctx, num_batches, use_amp, amp_dtype):
    model.eval()
    n = len(dataset)
    if n == 0:
        return 0.0
    losses = []
    for _ in range(num_batches):
        idxs = torch.randint(0, n, (batch_size,))
        x = torch.stack([dataset[int(i)][0] for i in idxs]).to(device)
        y = torch.stack([dataset[int(i)][1] for i in idxs]).to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            log_probs = model(x, None)
            losses.append(F.nll_loss(log_probs.view(-1, log_probs.size(-1)), y.view(-1)).item())
    model.train()
    return float(np.mean(losses))


def main():
    config = load_config(os.environ.get("CONFIG", "config.yaml"))
    seed = config.get("seed", 42)
    set_seed(seed)

    train_text = read_csv_text(config["data"]["train_text"])
    val_text = read_csv_text(config["data"]["val_text"])

    tokenizer = CharTokenizer()
    tokenizer.train(train_text + val_text)
    vocab_size = tokenizer.vocab_size

    s = {k: (float(v) if k in ("dropout", "weight_decay") else v) for k, v in config["structural"].items()}
    t = {k: (float(v) if k == "lr" else v) for k, v in config["training"].items()}
    seq_len = int(s["seq_len"])
    batch_size = int(t["batch_size"])
    max_steps = int(t.get("max_steps", 5000))
    base_lr = float(t["lr"])
    warmup_steps = int(t.get("warmup_steps", 100))
    weight_decay = float(s.get("weight_decay", 0.01))
    eval_interval = int(t.get("eval_interval", 200))
    sample_interval = int(t.get("sample_interval", 500))
    ckpt_interval = int(t.get("ckpt_interval", 1000))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} vocab_size={vocab_size}")

    train_dataset = ShakespeareDataset(tokenizer.encode(train_text), seq_len)
    val_dataset = ShakespeareDataset(tokenizer.encode(val_text), seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = build_decoder_only(config, vocab_size).to(device)
    raw_model = model
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params={n_params/1e6:.2f}M")

    use_amp = device.type == "cuda"
    if use_amp and not torch.cuda.is_bf16_supported():
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.bfloat16
    scaler = torch.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)
    print(f"amp={use_amp} dtype={amp_dtype}")

    if t.get("torch_compile", True) and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("torch.compile enabled")
        except Exception as e:
            print(f"torch.compile failed, continuing without: {e}")

    opt_cfg = {"AdamW": torch.optim.AdamW, "Adam": torch.optim.Adam}
    opt_cls = opt_cfg.get(t.get("optimizer", "AdamW"), torch.optim.AdamW)
    optimizer = opt_cls(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    experiment = None
    api_key = os.environ.get("COMET_API_KEY") or config.get("comet", {}).get("api_key", "")
    if COMET_AVAILABLE and api_key:
        experiment = Experiment(
            api_key=api_key,
            project_name=config["project_name"],
        )
        experiment.log_parameters({**s, **t, "vocab_size": vocab_size, "n_params": n_params})

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    jsonl_path = os.path.join(results_dir, "metrics.jsonl")

    def log_metrics(metrics, step):
        print(
            f"step={step} " + " ".join(f"{k}={v:.4g}" for k, v in metrics.items() if isinstance(v, float))
        )
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"step": step, **metrics}) + "\n")
        if experiment:
            experiment.log_metrics(metrics, step=step)

    model.train()
    step = 0
    best_val = float("inf")
    data_iter = iter(train_loader)

    while step < max_steps:
        for param_group in optimizer.param_groups:
            param_group["lr"] = get_lr(step, base_lr, warmup_steps, max_steps)

        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            log_probs = model(x, None)
            loss = F.nll_loss(
                log_probs.view(-1, log_probs.size(-1)), y.view(-1)
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step % 10 == 0:
            log_metrics({"train_loss": loss.item(), "lr": get_lr(step, base_lr, warmup_steps, max_steps)}, step)

        if step % eval_interval == 0 or step == max_steps - 1:
            val_loss = estimate_loss(model, val_dataset, device, batch_size, seq_len, 30, use_amp, amp_dtype)
            log_metrics({"val_loss": val_loss, "val_perplexity": math.exp(min(val_loss, 20))}, step)
            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {"model": raw_model.state_dict(), "step": step, "config": config, "vocab": tokenizer.vocab},
                    os.path.join(results_dir, "ckpt_best.pt"),
                )

        if step % sample_interval == 0 or step == max_steps - 1:
            raw_model.eval()
            ctx = x[:1, :64]
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                gen = raw_model.generate(ctx, max_new_tokens=t.get("sample_length", 256))
            raw_model.train()
            text = tokenizer.decode(gen[0].cpu().tolist())
            wf = word_fraction(tokenizer, gen[0].cpu().tolist())
            log_metrics({"word_fraction": wf}, step)
            sample_path = os.path.join(results_dir, f"sample_{step:06d}.txt")
            with open(sample_path, "w", encoding="utf-8") as f:
                f.write(text)
            if experiment:
                experiment.log_text(text, step=step)
            print(f"--- sample @ {step} (word_fraction={wf:.2f}) ---")
            print(text)

        if (step > 0 and step % ckpt_interval == 0) or step == max_steps - 1:
            ckpt = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "config": config,
                "vocab": tokenizer.vocab,
            }
            torch.save(ckpt, os.path.join(results_dir, f"ckpt_{step:06d}.pt"))

        step += 1

    if experiment:
        experiment.end()
    print("training done")


if __name__ == "__main__":
    main()
