import argparse

import torch

from decoder import build_decoder_only


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="results/ckpt_best.pt")
    parser.add_argument("--prompt", default="ROMEO:")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    vocab = ckpt["vocab"]

    model = build_decoder_only(config, len(vocab))
    model.load_state_dict(ckpt["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    torch.manual_seed(args.seed)
    inv_vocab = {v: k for k, v in vocab.items()}
    ids = [vocab.get(c, vocab["<unk>"]) for c in args.prompt]
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    out = model.generate(idx, max_new_tokens=args.max_new_tokens,
                         temperature=args.temperature, top_k=args.top_k)
    text = "".join(inv_vocab.get(int(i), "<UNK>") for i in out[0])
    print(text)


if __name__ == "__main__":
    main()
