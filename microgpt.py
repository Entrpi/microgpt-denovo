"""
microgpt.py

Single-file, pedagogical Python implementation of a tiny decoder-only GPT
trained on the makemore names corpus. The point is not framework cleverness;
it is to show the essential algorithmic pieces end to end in a compact,
readable form.

Model spec
- data: newline-delimited names from Karpathy's makemore corpus
- tokens: every unique character in the raw file, so '\n' is a real token
- architecture: 1 Transformer block, d_model=16, block_size=16, 4 heads
- GPT-2 spirit, but with RMSNorm, no biases, and ReLU instead of GeLU
- optimizer: Adam
- train: 1000 random next-token steps
- sample: start from '\n' and decode at temperature 0.5 until '\n'
"""

import urllib.request
import autograd.numpy as np
from autograd import grad


URL = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
T, C, H = 16, 16, 4          # block size, model width, attention heads
D, F = C // H, 4 * C         # head size, MLP hidden size
BATCH, STEPS = 64, 1000
LR, B1, B2, EPS = 3e-3, 0.9, 0.99, 1e-8
rng = np.random.RandomState(0)


def load_data():
    text = urllib.request.urlopen(URL).read().decode("utf-8")
    chars = sorted(set(text))                    # '\n' + lowercase alphabet
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    ids = np.array([stoi[ch] for ch in text], dtype=np.int32)
    return ids, stoi, itos, len(chars)


def init(shape, scale=0.02):
    fan_in = max(1, shape[0])
    return rng.randn(*shape) * (scale / np.sqrt(fan_in))


def init_params(vocab):
    return {
        "wte": init((vocab, C)),   # token embeddings, also tied output head
        "wpe": init((T, C)),       # learned positions 0..15
        "g1": np.ones(C),          # RMSNorm gain before attention
        "wq": init((C, C)), "wk": init((C, C)),
        "wv": init((C, C)), "wo": init((C, C)),
        "g2": np.ones(C),          # RMSNorm gain before MLP
        "fc": init((C, F)), "proj": init((F, C)),
        "gf": np.ones(C),          # final RMSNorm gain
    }


def zeros_like_tree(tree):
    return {k: np.zeros_like(v) for k, v in tree.items()}


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def logsumexp(x, axis=-1, keepdims=False):
    m = np.max(x, axis=axis, keepdims=True)
    y = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return y if keepdims else np.squeeze(y, axis=axis)


def rmsnorm(x, gain):
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-8)
    return gain * (x / rms)


def attention(x, p):
    b, t, _ = x.shape
    q = np.transpose((x @ p["wq"]).reshape(b, t, H, D), (0, 2, 1, 3))
    k = np.transpose((x @ p["wk"]).reshape(b, t, H, D), (0, 2, 1, 3))
    v = np.transpose((x @ p["wv"]).reshape(b, t, H, D), (0, 2, 1, 3))
    mask = np.triu(np.ones((t, t)), 1) * -1e9   # causal: forbid future tokens
    att = (q @ np.transpose(k, (0, 1, 3, 2))) / np.sqrt(D) + mask[None, None, :, :]
    y = softmax(att, axis=-1) @ v
    y = np.transpose(y, (0, 2, 1, 3)).reshape(b, t, C)
    return y @ p["wo"]


def forward(tokens, p):
    b, t = tokens.shape
    x = p["wte"][tokens] + p["wpe"][:t][None, :, :]
    x = x + attention(rmsnorm(x, p["g1"]), p)
    x = x + (np.maximum(0, rmsnorm(x, p["g2"]) @ p["fc"]) @ p["proj"])
    x = rmsnorm(x, p["gf"])
    return x @ p["wte"].T                      # tied output projection


def cross_entropy(logits, target, vocab):
    logp = logits - logsumexp(logits, axis=-1, keepdims=True)
    one_hot = np.eye(vocab)[target]           # pedagogical, not memory-optimal
    return -np.mean(np.sum(one_hot * logp, axis=-1))


def loss(p, x, y, vocab):
    return cross_entropy(forward(x, p), y, vocab)


def get_batch(ids):
    starts = rng.randint(0, len(ids) - T - 1, size=BATCH)
    x = np.stack([ids[i:i + T] for i in starts])
    y = np.stack([ids[i + 1:i + T + 1] for i in starts])
    return x, y


def adam_step(p, g, m, v, step):
    for k in p:
        m[k] = B1 * m[k] + (1 - B1) * g[k]
        v[k] = B2 * v[k] + (1 - B2) * (g[k] * g[k])
        m_hat = m[k] / (1 - B1 ** step)
        v_hat = v[k] / (1 - B2 ** step)
        p[k] = p[k] - LR * m_hat / (np.sqrt(v_hat) + EPS)
    return p, m, v


def sample(p, stoi, itos, temp=0.5, max_new_tokens=32):
    out = [stoi["\n"]]                        # reuse newline as BOS boundary
    for _ in range(max_new_tokens):
        ctx = np.array([out[-T:]], dtype=np.int32)
        logits = forward(ctx, p)[0, -1] / temp
        probs = softmax(logits)
        nxt = rng.choice(len(probs), p=np.asarray(probs))
        out.append(int(nxt))
        if itos[nxt] == "\n":
            break
    return "".join(itos[i] for i in out[1:-1])


def main():
    ids, stoi, itos, vocab = load_data()
    p = init_params(vocab)
    m, v = zeros_like_tree(p), zeros_like_tree(p)
    dloss = grad(lambda params, x, y: loss(params, x, y, vocab))

    for step in range(1, STEPS + 1):
        x, y = get_batch(ids)
        g = dloss(p, x, y)
        p, m, v = adam_step(p, g, m, v, step)
        if step % 100 == 0:
            print(f"step {step:4d} loss {loss(p, x, y, vocab):.4f}")

    for _ in range(20):
        print(sample(p, stoi, itos, temp=0.5))


if __name__ == "__main__":
    main()
