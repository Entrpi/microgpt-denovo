"""
Single-file, from-scratch Python implementation of a tiny decoder-only GPT
trained on the makemore names corpus. The goal is to show the algorithm as
directly as possible: a scalar autograd engine, list-based tensors, explicit
loops, and one coherent Transformer training loop.
"""

import math
import os
import random
import urllib.request

# =============================================================================
# 1. Setup, Scalar Autograd, Data, And Parameters
# =============================================================================
#
# We start with the scalar autograd engine itself: a tiny graph of `Value`
# objects that remembers how every number was formed. Then we load the names
# corpus, turn characters into token ids, and initialize the matrices that will
# become embeddings, attention, the MLP, and the output head.

URL = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"  # The tiny corpus we will learn from.
S = "\n"                                  # Newline marks name boundaries and also starts generation.
T, C, H = 16, 16, 4                       # Block size, model width, and number of attention heads.
LAYERS = int(os.getenv("MICROGPT_LAYERS", "1"))  # Stack this many identical Transformer blocks.
D, F = C // H, 4 * C                      # Per-head width and the hidden width of the MLP.
STEPS = 1000                              # Train for a small fixed number of updates.
LR, B1, B2, EPS = 3e-3, 0.9, 0.99, 1e-8   # Adam's step size, moment decay rates, and numerical floor.
TEMP, LOG_EVERY = 0.5, 100                # Sampling temperature and how often we print training progress.
NUM_SAMPLES, MAX_NEW = 20, T              # How many names to sample and how long each name may grow.
rng = random.Random(0)                    # A fixed seed keeps the demonstration repeatable.

class Value:                                                       # This is the scalar autograd engine that makes the whole file learn by hand.
    __slots__ = ("data", "grad", "_prev")                          # Each node keeps its value, its gradient, and edges of the form (child, local_grad).

    def __init__(self, data, prev=()):
        self.data, self.grad = float(data), 0.0                    # Each scalar stores its value and its accumulated gradient.
        self._prev = list(prev)                                    # Each edge already remembers the local derivative for the chain rule.

    def _coerce(self, other): return other if isinstance(other, Value) else Value(other)  # Python numbers are wrapped as leaves on demand.

    def __add__(self, other):                                      # Addition sends the upstream gradient unchanged to both inputs.
        other = self._coerce(other)
        return Value(self.data + other.data, ((self, 1.0), (other, 1.0)))
    __radd__ = __add__

    def __neg__(self): return Value(-self.data, ((self, -1.0),))   # Negation flips both the value and its local derivative.
    def __sub__(self, other): return self + (-other)               # Subtraction is addition of a negated term.
    def __rsub__(self, other): return other + (-self)

    def __mul__(self, other):                                      # Multiplication uses the product rule.
        other = self._coerce(other)
        return Value(self.data * other.data, ((self, other.data), (other, self.data)))
    __rmul__ = __mul__

    def __truediv__(self, other):                                  # Division uses the quotient rule.
        other = self._coerce(other)
        return Value(self.data / other.data, ((self, 1.0 / other.data), (other, -self.data / (other.data * other.data))))
    def __rtruediv__(self, other): return self._coerce(other) / self

    def exp(self):                                                 # Exponentials turn logits into positive mass for softmax.
        out = math.exp(self.data)
        return Value(out, ((self, out),))

    def log(self): return Value(math.log(self.data), ((self, 1.0 / self.data),))  # Logs turn probabilities into additive surprise.

    def sqrt(self):                                                # Square roots give RMSNorm its notion of scale.
        out = math.sqrt(self.data)
        return Value(out, ((self, 0.5 / out),))

    def relu(self): return Value(self.data if self.data > 0.0 else 0.0, ((self, 1.0 if self.data > 0.0 else 0.0),))  # ReLU is the MLP nonlinearity.

    def backward(self):                                            # Backward walks the graph in reverse topological order.
        topo, seen = [], set()                                     # Topological order guarantees each node receives complete downstream blame.
        def build(v):
            if v not in seen:
                seen.add(v)
                for child, _ in v._prev: build(child)
                topo.append(v)
        build(self)
        self.grad = 1.0                                            # The loss starts with unit gradient with respect to itself.
        for node in reversed(topo):
            for child, local_grad in node._prev:
                child.grad += local_grad * node.grad               # Chain rule: child.grad += local_grad * node.grad.

def load_data():                                                   # The tokenizer is just the unique character set of the corpus.
    text = urllib.request.urlopen(URL).read().decode("utf-8")
    chars = sorted(set(text))                                      # Every unique character becomes one token type.
    stoi = {ch: i for i, ch in enumerate(chars)}                   # Character-to-index lookup.
    itos = {i: ch for ch, i in stoi.items()}                       # Index-to-character lookup for decoding.
    ids = [stoi[ch] for ch in text]                                # The whole corpus becomes one long token stream.
    return ids, stoi, itos, len(chars)


def leaves(tree):                                                  # Walk every scalar leaf in a nested tensor tree.
    if isinstance(tree, dict):                                     # Gradients live only on scalar leaves, so we recurse through dicts and lists.
        for v in tree.values():
            yield from leaves(v)
    elif isinstance(tree, list):
        for v in tree:
            yield from leaves(v)
    else:
        yield tree

def zero_grads(tree):                                             # Backprop accumulates, so each step starts with fresh parameter gradients.
    for leaf in leaves(tree):
        leaf.grad = 0.0

def zeros_like(tree):                                             # Adam keeps its own tree-shaped moment buffers (with the same nested shape as the parameters)
    return {k: zeros_like(v) for k, v in tree.items()} if isinstance(tree, dict) else [zeros_like(v) for v in tree] if isinstance(tree, list) else 0.0  # Scalars start at zero.

def init_matrix(rows, cols, scale=0.02):                          # Small random weights keep the first residual stream near identity.
    fan_in = max(1, rows)
    std = scale / math.sqrt(fan_in)
    return [[Value(rng.gauss(0.0, std)) for _ in range(cols)] for _ in range(rows)]

def init_block(ones):                                             # A Transformer block is one attention branch plus one MLP branch.
    return {
        "g1": ones(C),                                            # RMSNorm scale before attention.
        "wq": init_matrix(C, C),                                  # Query projection.
        "wk": init_matrix(C, C),                                  # Key projection.
        "wv": init_matrix(C, C),                                  # Value projection.
        "wo": init_matrix(C, C),                                  # Output mix after the heads are merged.
        "g2": ones(C),                                            # RMSNorm scale before the MLP.
        "fc": init_matrix(C, F),                                  # MLP expansion layer.
        "proj": init_matrix(F, C),                                # MLP projection back to residual width.
    }

def init_params(vocab):                                           # One dict holds the whole model so the training loop can read it directly.
    ones = lambda n: [Value(1.0) for _ in range(n)]               # Norm gains start as neutral scalers.
    return {
        "wte": init_matrix(vocab, C),                             # Token embeddings, tied to the output head.
        "wpe": init_matrix(T, C),                                 # Learned position vectors for each visible slot.
        "blocks": [init_block(ones) for _ in range(LAYERS)],      # Stack as many identical blocks as the layer count requests.
        "gf": ones(C),                                            # Final RMSNorm scale before logits.
    }

# =============================================================================
# 2. Transformer Forward Pass
# =============================================================================
#
# This is the model's act of thinking. Tokens become vectors, attention lets
# each position read the causal past, the MLP refines those vectors, and the
# tied output head turns the final residual stream into next-character logits.

def linear(vec, mat):                                             # Matrix-vector multiply is the primitive behind every projection.
    out = []                                                      # A linear layer is just a bank of dot products, one row at a time.
    for row in mat:
        s = Value(0.0)
        for i in range(len(vec)):
            s = s + row[i] * vec[i]
        out.append(s)
    return out

def softmax(vec):                                                # Softmax turns logits into probabilities while staying numerically stable.
    m = max(v.data for v in vec)                                 # Shift by the largest logit before exponentiating to keep the arithmetic calm.
    exps = []
    denom = Value(0.0)                                           # The denominator is the total unnormalized mass after exponentiation.
    for v in vec:
        e = (v - m).exp()
        exps.append(e)
        denom = denom + e
    out = []
    for e in exps:
        out.append(e / denom)                                    # Divide each score by the total mass to get a proper probability.
    return out

def rmsnorm(vec, gain):                                          # RMSNorm rescales each token vector by its own magnitude, then restores scale.
    mean_sq = Value(0.0)
    for v in vec:
        mean_sq = mean_sq + v * v
    mean_sq = mean_sq / len(vec)                                 # Average squared magnitude across channels.
    scale = (mean_sq + EPS).sqrt()                               # Root-mean-square gives the vector's scale.
    out = []
    for i in range(len(vec)):
        out.append(gain[i] * (vec[i] / scale))
    return out

def attention(seq, block):                                       # Each token asks the causal past for the information it needs.
    n = len(seq)
    x = []
    for vec in seq:
        x.append(rmsnorm(vec, block["g1"]))                      # Normalize first so attention reads stable vectors.
    q, k, v = [], [], []                                         # The same vector becomes a question, a key, and a value.
    for vec in x:
        q.append(linear(vec, block["wq"]))                       # Queries ask what this position wants.
        k.append(linear(vec, block["wk"]))                       # Keys say what each past position offers.
        v.append(linear(vec, block["wv"]))                       # Values carry the content to be copied forward.
    out = []                                                     # One updated vector per position.
    for i in range(n):
        merged = [Value(0.0) for _ in range(C)]
        for h in range(H):                                       # Each head only sees the causal past of the current position.
            scores = []
            for j in range(i + 1):
                s = Value(0.0)
                for d in range(D):
                    idx = h * D + d
                    s = s + q[i][idx] * k[j][idx]
                scores.append(s / math.sqrt(D))                  # Scale dot products so softmax stays gentle.
            weights = softmax(scores)                            # Softmax turns scores into a distribution over visible positions.
            for d in range(D):
                idx = h * D + d
                acc = Value(0.0)
                for j in range(i + 1):
                    acc = acc + weights[j] * v[j][idx]
                merged[idx] = acc                                # Each head writes its own summary into its slice.
        out.append(linear(merged, block["wo"]))                  # The output projection recombines all heads into one update.
    return out

def mlp(seq, block):                                             # The MLP revises each position independently after attention has mixed context.
    out = []                                                     # Collect the per-position refinements.
    for vec in seq:
        h = rmsnorm(vec, block["g2"])                            # Normalize again before the feed-forward move.
        z = linear(h, block["fc"])
        relu = []
        for v in z:
            relu.append(v.relu())                                # ReLU keeps only the channels that fire.
        out.append(linear(relu, block["proj"]))                  # Project back to residual width.
    return out

def forward(tokens, p):                                          # One forward pass goes embeddings -> attention -> MLP -> tied logits.
    seq = []                                                     # The residual stream begins as embedded token vectors.
    for t in range(len(tokens)):
        vec = []
        for c in range(C):
            vec.append(p["wte"][tokens[t]][c] + p["wpe"][t][c])  # Content and position enter together.
        seq.append(vec)
    for block in p["blocks"]:                                    # Each layer repeats the same attention-then-MLP pattern.
        att = attention(seq, block)
        seq = [[seq[i][j] + att[i][j] for j in range(C)] for i in range(len(seq))]  # Residual stream keeps the old state plus the attention update.
        mid = mlp(seq, block)
        seq = [[seq[i][j] + mid[i][j] for j in range(C)] for i in range(len(seq))]  # The MLP adds another residual refinement.
    seq = [rmsnorm(vec, p["gf"]) for vec in seq]                                # Final normalization before vocabulary scoring.
    logits = []
    for vec in seq:
        logits.append(linear(vec, p["wte"]))
    return logits

# =============================================================================
# 3. Learning: Loss, Backward Pass, And Adam
# =============================================================================
#
# Now predictions become a learning signal. Cross-entropy asks how wrong the
# model was, the Value graph carries that scalar mistake backward through the
# computation, and Adam turns the gradients into small stable corrections.

def loss(tokens, targets, p):                                    # Cross-entropy rewards probability mass on the true next character.
    logits = forward(tokens, p)                                  # Training asks for high probability on the next correct character at each step.
    total = Value(0.0)                                           # Sum the surprise over the whole visible window.
    for t in range(len(targets)):
        probs = softmax(logits[t])                               # One position at a time becomes one probability distribution.
        total = total + (-probs[targets[t]].log())
    return total / len(targets)                                  # Average cross-entropy over the visible window.

def adam_step(tree, m, v, step):                                 # Adam keeps a running memory of direction and scale for every parameter.
    if isinstance(tree, dict):
        for k in tree:                                           # Recurse through named parameter groups so every scalar leaf gets updated.
            m[k], v[k] = adam_step(tree[k], m[k], v[k], step)
        return m, v
    if isinstance(tree, list):
        for i in range(len(tree)):                               # Recurse through nested lists in the same way.
            m[i], v[i] = adam_step(tree[i], m[i], v[i], step)
        return m, v
    g = tree.grad                                                # Each scalar leaf already knows its gradient.
    m = B1 * m + (1.0 - B1) * g                                  # First moment tracks the gradient direction.
    v = B2 * v + (1.0 - B2) * (g * g)                            # Second moment tracks gradient scale.
    mh = m / (1.0 - B1 ** step)                                  # Bias correction matters most early on.
    vh = v / (1.0 - B2 ** step)
    tree.data -= LR * mh / (math.sqrt(vh) + EPS)                 # Adam applies the step in place.
    return m, v


def sample_index(probs):                                         # Sample one token from a categorical distribution.
    u = rng.random()
    s = 0.0
    choice = len(probs) - 1                                      # Default to the final token if rounding leaves a gap.
    for i in range(len(probs)):
        s += probs[i].data
        if u <= s:
            choice = i
            break
    return choice

# =============================================================================
# 4. Training And Inference
# =============================================================================
#
# Here the whole algorithm runs as a process in time. Training repeatedly draws
# a short next-token problem, measures the model's mistake, sends that mistake
# backward through the graph, updates the weights, and then reuses the same
# forward pass to speak.

def main():
    ids, stoi, itos, vocab = load_data()
    p = init_params(vocab)
    m = zeros_like(p)                                            # Adam's first moment starts with no memory.
    v = zeros_like(p)                                            # Adam's second moment starts with no memory.

    for step in range(1, STEPS + 1):
        start = rng.randrange(0, len(ids) - T - 1)               # Pick a fresh slice of the corpus.
        tokens = ids[start:start + T]                            # The model sees the current characters.
        targets = ids[start + 1:start + T + 1]                   # The loss asks for the next characters.

        # --- Form The Loss -------------------------------------------------
        zero_grads(p)                                            # Old gradients are only for the last step.
        batch_loss = loss(tokens, targets, p)                    # The scalar loss is the single number we minimize.

        # --- Ask The Graph To Flow Backward -------------------------------
        batch_loss.backward()                                    # This is the autograd act: loss becomes gradients.

        # --- Let Adam Apply The Correction -------------------------------
        m, v = adam_step(p, m, v, step)                          # Adam turns gradients into new parameter values.

        if step % LOG_EVERY == 0:
            print(f"step {step:4d} loss {batch_loss.data:.4f}")  # Report the current training pressure.

    # --- Speak: Inference --------------------------------------------------
    for _ in range(NUM_SAMPLES):
        out = [stoi[S]]                                          # Every name begins at the boundary token.
        while len(out) <= MAX_NEW + 1:
            ctx = out[-T:]                                       # Only the visible context window matters.
            logits = forward(ctx, p)[-1]                         # Predict the next character from the last position.
            probs = softmax([z / TEMP for z in logits])          # Temperature makes sampling less or more adventurous.
            nxt = sample_index(probs)                            # Draw the next character from the distribution.
            out.append(nxt)
            if itos[nxt] == S:
                break                                            # Newline ends the sampled name.
        print("".join(itos[i] for i in out[1:-1]))               # Strip boundary tokens before printing.

if __name__ == "__main__":
    main()
