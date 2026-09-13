import csv
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

csv.field_size_limit(sys.maxsize)


def read_csv_text(path: str) -> str:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [row["text"] for row in reader if row.get("text")]
    return "\n".join(rows)


class CharTokenizer:
    def __init__(self):
        self.vocab = {"<unk>": 0}
        self.inverse_vocab = {0: "<unk>"}

    def train(self, text: str):
        for char in sorted(set(text)):
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str):
        return [self.vocab.get(char, self.vocab["<unk>"]) for char in text]

    def decode(self, ids):
        return "".join(self.inverse_vocab.get(int(i), "<UNK>") for i in ids)

    @property
    def vocab_size(self):
        return len(self.vocab)


class ShakespeareDataset(Dataset):
    def __init__(self, encoded_text, seq_len):
        self.data = torch.tensor(encoded_text, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y


class Embeddings(nn.Module):
    def __init__(self, vocab_size, seq_len, d_model, dropout=0.0):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x, pos_offset=0):
        seq_len_current = x.size(1)
        tok_emb = self.token_embed(x) * math.sqrt(self.d_model)
        pos_emb = self.pe[:, pos_offset : pos_offset + seq_len_current, :]
        return self.dropout(tok_emb + pos_emb)


class MultiHeadAttention(nn.Module):
    """MHA / GQA / MQA attention.

    n_kv_heads == n_heads  -> MHA (classic, Vaswani et al. 2017)
    n_kv_heads == 1       -> MQA (Shazeer, 2019)
    1 < n_kv_heads < n_heads -> GQA (Ainslie et al., 2023)

    use_sdpa=True -> torch.nn.functional.scaled_dot_product_attention, which
    dispatches to FlashAttention / memory-efficient kernels on GPU.
    use_sdpa=False -> classic manual attention (матрица скоров в явном виде).
    """

    def __init__(self, d_model, n_heads, dropout, n_kv_heads=None, use_sdpa=False):
        super().__init__()
        n_kv_heads = n_kv_heads or n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.d_model = d_model
        self.h = n_heads
        self.n_kv = n_kv_heads
        self.group = n_heads // n_kv_heads
        self.d_k = d_model // n_heads
        self.use_sdpa = use_sdpa
        self.dropout_p = float(dropout)

        self.W_q = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_o = nn.Linear(n_heads * self.d_k, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, cache=None):
        """x: (B, T, D). cache: tuple (k, v) прошлых шагов, (B, n_kv, S, d_k).

        Возвращает (out, new_cache). new_cache хранит НЕ растянутые k/v
        (n_kv голов) - в этом и есть выигрыш GQA/MQA по памяти KV-кэша.
        """
        B, T, _ = x.shape

        q = self.W_q(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_kv, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_kv, self.d_k).transpose(1, 2)

        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)

        # KV-кэш храним нерастянутым (n_kv голов) - в этом выигрыш GQA/MQA
        new_cache = (k, v)

        if self.group > 1:
            k = k.repeat_interleave(self.group, dim=1)
            v = v.repeat_interleave(self.group, dim=1)

        if self.use_sdpa:
            # FlashAttention / memory-efficient kernel (IO-aware, tiling).
            # is_causal применим только когда запросы и ключи одной длины
            # (префилл/тренировка); на шаге декодирования (T == 1) запрос
            # legitimately смотрит на весь кэш - маска не нужна.
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=(T > 1 and cache is None),
            )
            return self.W_o(out.transpose(1, 2).contiguous().view(B, T, -1)), new_cache

        # Manual attention (как в оригинальном трансформере)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = scores.softmax(dim=-1)
        scores = self.dropout(scores)
        out = scores @ v
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(out), new_cache


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return ((x - mean) / torch.sqrt(var + self.eps)) * self.alpha + self.bias


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MoEFeedForward(nn.Module):
    """DeepSeek-style Mixture-of-Experts FFN (DeepSeekMoE + балансировка V3).

    h_t' = u_t + sum_i FFN_i^shared(u_t) + sum_i g_i,t * FFN_i^routed(u_t)

    Роутинг (V3): affinity = sigmoid(u e_i), top-K по (affinity + bias),
    гейты = выбранные affinity, нормированные на их сумму (bias в гейт не входит).

    Два режима балансировки:
      - "aux_free": auxiliary-loss-free (Wang et al. 2024, DeepSeek-V3) -
        bias b_i обновляется на каждом шаге: перегруженным экспертам bias
        уменьшаем, недогруженным - увеличиваем (update_bias, вызывает trainer
        после optimizer step). Плюс крошечный sequence-wise balance loss
        (V3, alpha=1e-4), чтобы избежать экстремального дисбаланса внутри
        одной последовательности.
      - "aux_loss": классический aux loss (Switch Transformer / GShard) -
        L = alpha * N_r * sum_i f_i P_i на softmax-роутере.
    """

    def __init__(
        self,
        d_model,
        dropout: float,
        d_ff_shared=768,
        d_ff_expert=384,
        n_shared=1,
        n_routed=8,
        top_k=2,
        balance="aux_free",
        balance_alpha=1e-4,
        bias_gamma=1e-3,
    ):
        super().__init__()
        self.n_routed = n_routed
        self.n_shared = n_shared
        self.top_k = top_k
        self.balance = balance
        self.balance_alpha = float(balance_alpha)
        self.bias_gamma = float(bias_gamma)

        self.router = nn.Linear(d_model, n_routed, bias=False)
        self.shared = FeedForward(d_model, d_ff_shared, dropout) if n_shared > 0 else None
        self.experts = nn.ModuleList(
            [FeedForward(d_model, d_ff_expert, dropout) for _ in range(n_routed)]
        )
        self.register_buffer("expert_bias", torch.zeros(n_routed))
        self.register_buffer("expert_count", torch.zeros(n_routed))
        self.last_balance_loss = None
        self.last_load_std = None

    def forward(self, x):
        B, T, D = x.shape
        N = B * T
        u = x.reshape(N, D)

        router_logits = self.router(u)
        if self.balance == "aux_loss":
            aff = router_logits.softmax(dim=-1)
        else:
            aff = torch.sigmoid(router_logits)

        if self.balance == "aux_free":
            # top-K выбираем по affinity + bias, гейт - по "чистому" affinity
            route_scores = aff + self.expert_bias
        else:
            route_scores = aff
        topk_scores, topk_idx = route_scores.topk(self.top_k, dim=-1)

        sel_aff = aff.gather(1, topk_idx)
        gate = sel_aff / sel_aff.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        if self.shared is not None:
            out = self.shared(u)
        else:
            out = torch.zeros_like(u)

        for e, expert in enumerate(self.experts):
            sel = topk_idx == e
            rows = sel.any(dim=-1)
            if rows.any():
                idx = rows.nonzero(as_tuple=True)[0]
                w = (gate * sel).sum(dim=-1)[idx].unsqueeze(-1)
                y = expert(u[idx])
                out = out.index_add(0, idx, y * w)

        if self.balance == "aux_free":
            # Sequence-wise balance loss (V3, eq. 17-20), в среднем по батчу
            s_norm = aff / aff.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            P = s_norm.view(B, T, self.n_routed).mean(dim=1)  # (B, E)
            one_hot = F.one_hot(topk_idx, self.n_routed)  # (N, K, E)
            counts = one_hot.sum(dim=1).view(B, T, self.n_routed).sum(dim=1)  # (B, E)
            f = counts * (self.n_routed / (self.top_k * T))
            self.last_balance_loss = self.balance_alpha * (f * P).sum(dim=-1).mean()
        else:
            # Switch/GShard-style: L = alpha * N_r * sum_i f_i * P_i
            P = aff.mean(dim=0)
            f = F.one_hot(topk_idx, self.n_routed).sum(dim=(0, 1)).float()
            f = f * (self.n_routed / (N * self.top_k))
            self.last_balance_loss = self.balance_alpha * self.n_routed * (f * P).sum()

        with torch.no_grad():
            counts = F.one_hot(topk_idx, self.n_routed).sum(dim=(0, 1)).float()
            self.expert_count.copy_(counts)
            frac = counts / counts.sum().clamp_min(1.0)
            self.last_load_std = frac.std().item()

        return out.view(B, T, D)

    @torch.no_grad()
    def update_bias(self):
        """Auxiliary-loss-free балансировка (DeepSeek-V3): bias-коррекция
        перегруженных/недогруженных экспертов с шагом bias_gamma."""
        if self.balance != "aux_free" or self.expert_count.sum() == 0:
            return
        mean = self.expert_count.mean()
        self.expert_bias += self.bias_gamma * torch.sign(mean - self.expert_count)


class ResidualConnection(nn.Module):
    def __init__(self, d_model, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(d_model)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))  # Post LN


class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block, feed_forward_block, d_model, dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList(
            [ResidualConnection(d_model, dropout) for _ in range(2)]
        )

    def forward(self, x, mask, cache=None):
        attn_out, new_cache = self.self_attention_block(x, mask, cache)
        x = self.residual_connection[0](x, lambda _z: attn_out)
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x, new_cache


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, d_model):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)


class DecoderOnly(nn.Module):
    def __init__(self, embeddings, decoder, projection_layer):
        super().__init__()
        self.embeddings = embeddings
        self.decoder = decoder
        self.projection_layer = projection_layer
        seq_len = embeddings.pe.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len)
        self.register_buffer("causal_mask", mask)

    def forward(self, x, mask=None, caches=None, pos_offset=0, want_cache=False):
        """Возвращает (log_probs, caches). caches - список (k, v) по слоям;
        заполняется, если want_cache=True (генерация) или передан caches."""
        seq_len = x.size(1)
        if mask is None and seq_len > 1 and caches is None:
            mask = self.causal_mask[:, :, :seq_len, :seq_len]
        x = self.embeddings(x, pos_offset)
        new_caches = []
        for i, layer in enumerate(self.decoder.layers):
            c_in = caches[i] if caches is not None else None
            x, c_out = layer(x, mask, c_in)
            new_caches.append(c_out)
        x = self.decoder.norm(x)
        return self.projection_layer(x), (new_caches if (want_cache or caches is not None) else None)

    def get_moe_loss(self):
        total = None
        for m in self.modules():
            if isinstance(m, MoEFeedForward) and m.last_balance_loss is not None:
                total = m.last_balance_loss if total is None else total + m.last_balance_loss
        return total

    @torch.no_grad()
    def update_moe_bias(self):
        for m in self.modules():
            if isinstance(m, MoEFeedForward):
                m.update_bias()

    def moe_load_std(self):
        vals = [
            m.last_load_std
            for m in self.modules()
            if isinstance(m, MoEFeedForward) and m.last_load_std is not None
        ]
        return sum(vals) / len(vals) if vals else None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50, use_cache=True):
        max_seq_len = self.embeddings.pe.size(1)
        caches = None

        for i in range(max_new_tokens):
            if use_cache:
                if i == 0:
                    inp = idx if idx.size(1) <= max_seq_len else idx[:, -max_seq_len:]
                    pos_offset = 0
                else:
                    cl = caches[0][0].size(2)
                    if cl >= max_seq_len:
                        caches = [
                            (k[:, :, -(max_seq_len - 1) :], v[:, :, -(max_seq_len - 1) :])
                            for k, v in caches
                        ]
                        cl = max_seq_len - 1
                    inp = idx[:, -1:]
                    pos_offset = cl
                log_probs, caches = self.forward(inp, None, caches, pos_offset, want_cache=True)
                log_probs = log_probs[:, -1, :]
            else:
                idx_cond = idx if idx.size(1) <= max_seq_len else idx[:, -max_seq_len:]
                log_probs, _ = self.forward(idx_cond, None)
                log_probs = log_probs[:, -1, :]

            log_probs = log_probs / max(temperature, 1e-5)
            if top_k:
                kth = torch.topk(log_probs, top_k, dim=-1).values[..., -1:]
                log_probs[log_probs < kth] = float("-inf")
            probs = log_probs.softmax(dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx


def _init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


def build_decoder_only(config, vocab_size):
    s = config["structural"]
    d_model = s["d_model"]
    n_layers = s["n_layers"]
    n_heads = s["n_heads"]
    d_ff = s["d_ff"]
    seq_len = s["seq_len"]
    dropout = float(s["dropout"])

    att_type = s.get("attention", "mha")
    n_kv = s.get("n_kv_heads", None) or n_heads
    if att_type == "mha":
        n_kv = n_heads
    elif att_type == "mqa":
        n_kv = 1
    elif att_type != "gqa":
        raise ValueError(f"unknown attention type: {att_type}")
    use_sdpa = bool(s.get("use_sdpa", False))

    ffn_type = s.get("ffn", "dense")
    moe_cfg = s.get("moe", {}) or {}

    embeddings = Embeddings(vocab_size, seq_len, d_model, dropout)
    blocks = []
    for _ in range(n_layers):
        if ffn_type == "moe":
            ffn_block = MoEFeedForward(
                d_model,
                dropout,
                d_ff_shared=int(moe_cfg.get("d_ff_shared", 768)),
                d_ff_expert=int(moe_cfg.get("d_ff", 384)),
                n_shared=int(moe_cfg.get("n_shared", 1)),
                n_routed=int(moe_cfg.get("n_routed", 8)),
                top_k=int(moe_cfg.get("top_k", 2)),
                balance=moe_cfg.get("balance", "aux_free"),
                balance_alpha=float(moe_cfg.get("balance_alpha", 1e-4)),
                bias_gamma=float(moe_cfg.get("bias_gamma", 1e-3)),
            )
        else:
            ffn_block = FeedForward(d_model, d_ff, dropout)
        blocks.append(
            DecoderBlock(
                MultiHeadAttention(d_model, n_heads, dropout, n_kv_heads=n_kv, use_sdpa=use_sdpa),
                ffn_block,
                d_model,
                dropout,
            )
        )
    decoder = Decoder(nn.ModuleList(blocks), d_model)
    projection_layer = ProjectionLayer(d_model, vocab_size)

    model = DecoderOnly(embeddings, decoder, projection_layer)
    model.apply(_init_weights)
    return model


def count_params(model):
    """(total, active): active - параметры, участвующие в вычислении для
    одного токена (для MoE: shared + top_k экспертов вместо всех n_routed)."""
    total = sum(p.numel() for p in model.parameters())
    inactive = 0.0
    for m in model.modules():
        if isinstance(m, MoEFeedForward):
            expert_params = sum(p.numel() for p in m.experts.parameters())
            inactive += expert_params * (m.n_routed - m.top_k) / m.n_routed
    return total, total - inactive
