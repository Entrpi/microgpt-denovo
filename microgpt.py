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


# =============================================================================
# 1. Setup, Data, And Parameters
# =============================================================================

URL = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
S = "\n"                     # Newline is the only boundary token in the stream.
T, C, H = 16, 16, 4          # block size, model width, attention heads
D, F = C // H, 4 * C         # head size, MLP hidden size
BATCH, STEPS = 64, 1000
LR, B1, B2, EPS = 3e-3, 0.9, 0.99, 1e-8
TEMP, LOG_EVERY = 0.5, 100
NUM_SAMPLES, MAX_NEW = 20, T
rng = np.random.RandomState(0)


def load_data():
    text = urllib.request.urlopen(URL).read().decode("utf-8")
    chars = sorted(set(text))                    # Use exactly the characters that appear in the corpus.
    stoi = {ch: i for i, ch in enumerate(chars)} # Character -> integer token id.
    itos = {i: ch for ch, i in stoi.items()}     # Integer token id -> character.
    ids = np.array([stoi[ch] for ch in text], dtype=np.int32)  # Encode the whole corpus once as a stream.
    return ids, stoi, itos, len(chars)

def init(shape, scale=0.02):
    fan_in = max(1, shape[0])
    return rng.randn(*shape) * (scale / np.sqrt(fan_in))


def init_params(vocab):
    return {
        "wte": init((vocab, C)),   # Token embeddings, also reused as the tied output head.
        "wpe": init((T, C)),       # Position embeddings for visible slots 0..T-1.
        "g1": np.ones(C),          # RMSNorm gain before attention.
        "wq": init((C, C)), "wk": init((C, C)),
        "wv": init((C, C)), "wo": init((C, C)),
        "g2": np.ones(C),          # RMSNorm gain before the feed-forward MLP.
        "fc": init((C, F)), "proj": init((F, C)),
        "gf": np.ones(C),          # Final RMSNorm gain before logits.
    }


def zeros_like_tree(tree):
    return {k: np.zeros_like(v) for k, v in tree.items()}


# =============================================================================
# 2. Transformer Forward Pass
# =============================================================================

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)  # Shift logits for numerical stability.
    ex = np.exp(x)                               # Exponentiate the centered logits.
    return ex / np.sum(ex, axis=axis, keepdims=True)


def logsumexp(x, axis=-1, keepdims=False):
    m = np.max(x, axis=axis, keepdims=True)      # Pull out the largest logit before exponentiating.
    y = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return y if keepdims else np.squeeze(y, axis=axis)


def rmsnorm(x, gain):
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-8)  # Root-mean-square size of each token vector.
    return gain * (x / rms)


def attention(x, p):
    b, t, _ = x.shape
    q = np.transpose((x @ p["wq"]).reshape(b, t, H, D), (0, 2, 1, 3))   # Queries ask what each position wants.
    k = np.transpose((x @ p["wk"]).reshape(b, t, H, D), (0, 2, 1, 3))   # Keys say what each visible position offers.
    v = np.transpose((x @ p["wv"]).reshape(b, t, H, D), (0, 2, 1, 3))   # Values carry the content to be copied forward.
    mask = np.triu(np.ones((t, t)), 1) * -1e9                           # Causal mask forbids looking into the future.
    att = (q @ np.transpose(k, (0, 1, 3, 2))) / np.sqrt(D) + mask[None, None, :, :]
    y = softmax(att, axis=-1) @ v                                       # Attend to past values with softmax-normalized scores.
    y = np.transpose(y, (0, 2, 1, 3)).reshape(b, t, C)                  # Recombine the heads into one residual-width vector.
    return y @ p["wo"]                                                  # Mix the concatenated heads back into model space.


def forward(tokens, p):
    b, t = tokens.shape
    x = p["wte"][tokens] + p["wpe"][:t][None, :, :]        # Add token identity and token position.
    x = x + attention(rmsnorm(x, p["g1"]), p)              # Attention reads from the causal past, then updates the residual stream.
    x = x + (np.maximum(0, rmsnorm(x, p["g2"]) @ p["fc"]) @ p["proj"])  # The MLP refines each position independently.
    x = rmsnorm(x, p["gf"])                                # Final normalization before vocabulary scoring.
    return x @ p["wte"].T                                  # Tie output logits back to the embedding table.


# =============================================================================
# 3. Learning: Loss, Gradients, And Adam
# =============================================================================

def cross_entropy(logits, target, vocab):
    logp = logits - logsumexp(logits, axis=-1, keepdims=True)  # Convert logits into log-probabilities.
    one_hot = np.eye(vocab)[target]                            # Build pedagogical one-hot targets.
    return -np.mean(np.sum(one_hot * logp, axis=-1))


def loss(p, x, y, vocab):
    logits = forward(x, p)                                 # The forward pass turns each visible context into next-token scores.
    return cross_entropy(logits, y, vocab)                 # The loss asks whether the true next characters received high probability.

def learning_signal(vocab):
    objective = lambda params, x, y: loss(params, x, y, vocab)     # Differentiate the training objective we just defined above.
    return grad(objective)                                         # This backward function tells every weight how to reduce the loss.

def get_batch(ids):
    starts = rng.randint(0, len(ids) - T - 1, size=BATCH)     # Choose random windows in the long character stream.
    x = np.stack([ids[i:i + T] for i in starts])              # Inputs are length-T contexts.
    y = np.stack([ids[i + 1:i + T + 1] for i in starts])      # Targets are those same windows shifted one token ahead.
    return x, y


def adam_step(p, g, m, v, step):
    for k in p:
        m[k] = B1 * m[k] + (1 - B1) * g[k]                    # Update Adam's running average of gradients.
        v[k] = B2 * v[k] + (1 - B2) * (g[k] * g[k])           # Update Adam's running average of squared gradients.
        m_hat = m[k] / (1 - B1 ** step)                       # Bias-correct the first moment.
        v_hat = v[k] / (1 - B2 ** step)                       # Bias-correct the second moment.
        p[k] = p[k] - LR * m_hat / (np.sqrt(v_hat) + EPS)     # Take the adaptive parameter step.
    return p, m, v


# =============================================================================
# 4. Training And Inference
# =============================================================================

def main():
    ids, stoi, itos, vocab = load_data()                            # Load and tokenize the raw character stream.
    p = init_params(vocab)                                          # Initialize the tiny GPT weights.
    m, v = zeros_like_tree(p), zeros_like_tree(p)                   # Adam moment buffers start at zero.
    backward = learning_signal(vocab)                               # This is the model's backward pass: it turns loss into gradients.

    for step in range(1, STEPS + 1):
        x, y = get_batch(ids)                                       # Draw a fresh minibatch of next-token problems.

        # --- Form The Loss -------------------------------------------------
        batch_loss = loss(p, x, y, vocab)                           # First measure how surprised the model is by the true next characters.

        # --- Ask For Gradients --------------------------------------------
        g = backward(p, x, y)                                       # Then ask autograd how every parameter should move to reduce that surprise.

        # --- Let Adam Apply The Correction -------------------------------
        p, m, v = adam_step(p, g, m, v, step)                       # Adam turns that learning signal into the actual parameter update.

        if step % LOG_EVERY == 0:
            print(f"step {step:4d} loss {batch_loss:.4f}")

    # --- Speak: Inference --------------------------------------------------

    for _ in range(NUM_SAMPLES):
        out = [stoi[S]]                                             # Start from the boundary token that marks a new name.
        for _ in range(MAX_NEW):
            ctx = np.array([out[-T:]], dtype=np.int32)              # Only the most recent T tokens are visible.
            logits = forward(ctx, p)[0, -1] / TEMP                  # Predict the next-token logits for the last visible position.
            probs = softmax(logits)                                 # Turn logits into a categorical distribution.
            nxt = rng.choice(len(probs), p=np.asarray(probs))       # Sample instead of argmax so names vary across runs.
            out.append(int(nxt))
            if itos[nxt] == S:
                break
        print("".join(itos[i] for i in out[1:-1]))


if __name__ == "__main__":
    main()
