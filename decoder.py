import csv
import math
import sys

import torch
import torch.nn as nn
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

    def forward(self, x):
        seq_len_current = x.size(1)
        tok_emb = self.token_embed(x) * math.sqrt(self.d_model)
        pos_emb = self.pe[:, :seq_len_current, :]
        return self.dropout(tok_emb + pos_emb)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.h = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.W_q(q).view(q.size(0), q.size(1), self.h, self.d_k).transpose(1, 2)
        key = self.W_k(k).view(k.size(0), k.size(1), self.h, self.d_k).transpose(1, 2)
        value = self.W_v(v).view(v.size(0), v.size(1), self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttention.attention(
            query, key, value, mask, self.dropout
        )

        x = x.transpose(1, 2).contiguous().view(x.size(0), x.size(2), self.h * self.d_k)
        return self.W_o(x)


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return ((x - mean) / torch.sqrt(std + self.eps)) * self.alpha + self.bias


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


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

    def forward(self, x, mask):
        x = self.residual_connection[0](
            x, lambda z: self.self_attention_block(z, z, z, mask)
        )
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x


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

    def forward(self, x, mask=None):
        seq_len = x.size(1)
        if mask is None:
            mask = self.causal_mask[:, :, :seq_len, :seq_len]
        x = self.embeddings(x)
        x = self.decoder(x, mask)
        return self.projection_layer(x)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50):
        max_seq_len = self.embeddings.pe.size(1)
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= max_seq_len else idx[:, -max_seq_len:]
            log_probs = self.forward(idx_cond, None)[:, -1, :]
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
    dropout = s["dropout"]

    embeddings = Embeddings(vocab_size, seq_len, d_model, dropout)
    blocks = [
        DecoderBlock(
            MultiHeadAttention(d_model, n_heads, dropout),
            FeedForward(d_model, d_ff, dropout),
            d_model,
            dropout,
        )
        for _ in range(n_layers)
    ]
    decoder = Decoder(nn.ModuleList(blocks), d_model)
    projection_layer = ProjectionLayer(d_model, vocab_size)

    model = DecoderOnly(embeddings, decoder, projection_layer)
    model.apply(_init_weights)
    return model
