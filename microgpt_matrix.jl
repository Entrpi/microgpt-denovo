# Single-file, from-scratch Julia implementation of a tiny decoder-only GPT
# trained on the makemore names corpus, written as explicit matrix calculus.
# Where the sister file `microgpt.py` records derivatives automatically with a
# scalar autograd engine, here every backward pass is derived by hand: each
# forward stage saves its intermediates onto a tape, and learning replays that
# tape in reverse. The journey: data and weights, the forward pass, the
# hand-written backward pass and Adam, and finally the trained model speaking.

using Downloads                                                     # Standard library download support keeps the script self-contained.
using LinearAlgebra                                                 # We need transposes and dense matrix multiplication.
using Random                                                        # Training and sampling both rely on randomness.
using Statistics                                                    # RMSNorm and cross-entropy use reductions like mean.

# =============================================================================
# 1. Setup, Data, And Parameters
# =============================================================================
#
# We begin by deciding what tiny GPT we want to study. Then we turn the raw names
# file into one stream of integer tokens and create the tensors that will hold
# the model's memory: embeddings, attention projections, MLP weights, and gains.

const URL = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"  # Train on the canonical makemore names file.
const S = '\n'                                                      # Newline is the only boundary token in the stream.
const T = 16                                                        # The model sees at most 16 previous characters.
const C = 16                                                        # Residual width and embedding width are both 16.
const H = 4                                                         # Use 4 attention heads.
const LAYERS = parse(Int, get(ENV, "MICROGPT_LAYERS", "1"))         # Stack this many identical Transformer blocks.
const D = C ÷ H                                                     # Each head gets width 4.
const F = 4 * C                                                     # The MLP widens channels by the usual GPT factor of 4.
const BATCH = 64                                                    # Train on 64 random windows per step.
const STEPS = parse(Int, get(ENV, "MICROGPT_STEPS", "1000"))        # Allow fast smoke tests without changing the default spec.
const LR = 3.0f-3                                                   # Adam base learning rate.
const B1 = 0.9f0                                                    # Adam first-moment decay.
const B2 = 0.99f0                                                   # Adam second-moment decay.
const EPS = 1.0f-8                                                  # Numerical floor for RMSNorm and Adam.
const TEMP = parse(Float32, get(ENV, "MICROGPT_TEMP", "0.5"))       # Sample with the requested 0.5 temperature by default.
const LOG_EVERY = parse(Int, get(ENV, "MICROGPT_LOG_EVERY", "100")) # Print loss occasionally during training.
const NUM_SAMPLES = parse(Int, get(ENV, "MICROGPT_SAMPLES", "20"))  # Emit several names after training.
const MAX_NEW = T                                                   # Do not sample longer than the full context window.
const SEED = 0                                                      # Keep runs repeatable.
const MASK = triu(fill(-1.0f9, T, T), 1)                            # One fixed causal mask: the upper triangle hides the future.

function load_data()                                                # Read the corpus and build the minimal character tokenizer.
    text = read(Downloads.download(URL), String)                    # Treat the entire file as one long stream of characters.
    itos = sort(collect(Set(text)))                                 # Use exactly the unique characters that appear in the data.
    stoi = Dict{Char, Int}(ch => i for (i, ch) in enumerate(itos))  # Map chars to integer token ids.
    ids = Int[stoi[ch] for ch in text]                              # Encode the full corpus once for efficient random slicing.
    return ids, stoi, itos                                          # `itos[i]` is the inverse map from token id back to char.
end                                                                 # This keeps tokenization maximally simple.

mutable struct BlockParams                                          # One Transformer block is one attention branch plus one MLP branch.
    g1::Vector{Float32}                                             # RMSNorm gain before attention.
    Wq::Matrix{Float32}                                             # Query projection.
    Wk::Matrix{Float32}                                             # Key projection.
    Wv::Matrix{Float32}                                             # Value projection.
    Wo::Matrix{Float32}                                             # Attention output projection.
    g2::Vector{Float32}                                             # RMSNorm gain before the MLP.
    W1::Matrix{Float32}                                             # MLP expansion matrix.
    W2::Matrix{Float32}                                             # MLP contraction matrix.
end                                                                 # Stacking layers means stacking these block parameter bundles.

mutable struct Params                                               # The whole model is embeddings, a stack of blocks, and one final norm.
    E::Matrix{Float32}                                              # Token embeddings, also reused as the tied output head.
    P::Matrix{Float32}                                              # Position embeddings for slots 1..T.
    blocks::Vector{BlockParams}                                     # Repeat the same block structure as many times as LAYERS requests.
    gf::Vector{Float32}                                             # Final RMSNorm gain before logits.
end                                                                 # This is still a tiny GPT, just no longer hardcoded to depth 1.

const BLOCK_NAMES = fieldnames(BlockParams)                         # Adam will iterate over block-local tensors uniformly.

function init_block(rng)                                            # Initialize one Transformer block with small random weights.
    return BlockParams(                                             # Weights start at scale 1/sqrt(fan_in) so early activations stay O(1).
        ones(Float32, C),                                           # RMSNorm gains start as identity scales.
        randn(rng, Float32, C, C) / sqrt(Float32(C)),               # Query matrix.
        randn(rng, Float32, C, C) / sqrt(Float32(C)),               # Key matrix.
        randn(rng, Float32, C, C) / sqrt(Float32(C)),               # Value matrix.
        randn(rng, Float32, C, C) / sqrt(Float32(C)),               # Attention output matrix.
        ones(Float32, C),                                           # Second RMSNorm gain.
        randn(rng, Float32, C, F) / sqrt(Float32(C)),               # MLP expansion.
        randn(rng, Float32, F, C) / sqrt(Float32(F)),               # MLP contraction.
    )                                                               # One block is now fully initialized.
end                                                                 # Stacking layers just repeats this same recipe.

function init_params(rng, V)                                        # Initialize the whole tiny GPT.
    return Params(
        randn(rng, Float32, V, C) / sqrt(Float32(C)),               # Token embeddings.
        randn(rng, Float32, T, C) / sqrt(Float32(C)),               # Position embeddings.
        [init_block(rng) for _ in 1:LAYERS],                        # One parameter bundle per Transformer layer.
        ones(Float32, C),                                           # Final RMSNorm gain.
    )                                                               # This is enough to start training immediately.
end                                                                 # No biases are used anywhere, by design.

function zeros_like(b::BlockParams)                                 # Adam moment buffers match each block's shapes exactly.
    return BlockParams(
        zeros(Float32, size(b.g1)...),                              # First RMSNorm gain buffer.
        zeros(Float32, size(b.Wq)...),                              # Query buffer.
        zeros(Float32, size(b.Wk)...),                              # Key buffer.
        zeros(Float32, size(b.Wv)...),                              # Value buffer.
        zeros(Float32, size(b.Wo)...),                              # Attention output buffer.
        zeros(Float32, size(b.g2)...),                              # Second RMSNorm gain buffer.
        zeros(Float32, size(b.W1)...),                              # MLP expansion buffer.
        zeros(Float32, size(b.W2)...),                              # MLP contraction buffer.
    )                                                               # Every block tensor gets a zero-filled twin.
end                                                                 # This keeps Adam explicit even with multiple layers.

function zeros_like(p::Params)                                      # Adam also needs top-level buffers for embeddings and the final norm.
    return Params(
        zeros(Float32, size(p.E)...),                               # Token embedding buffer.
        zeros(Float32, size(p.P)...),                               # Position embedding buffer.
        [zeros_like(b) for b in p.blocks],                          # One block buffer per Transformer layer.
        zeros(Float32, size(p.gf)...),                              # Final RMSNorm gain buffer.
    )                                                               # The buffer tree mirrors the parameter tree exactly.
end                                                                 # This is enough for Adam to update any layer depth.

# =============================================================================
# 2. Transformer Forward Pass
# =============================================================================
#
# This section is the model's act of thinking. Tokens become vectors, attention
# lets each position read from the causal past, the MLP reshapes that information,
# and the tied output head turns the residual stream into next-character logits.
# Because we will differentiate everything by hand, each stage also returns a
# cache of its intermediates: the tape that learning will later replay in reverse.

function linear(X, W)                                               # Apply a weight matrix to the last dimension of a 3D tensor.
    X2 = reshape(X, :, size(X, 3))                                  # Collapse time and batch so one matmul handles every token vector.
    return reshape(X2 * W, size(X, 1), size(X, 2), size(W, 2))
end                                                                 # This is the basic projection primitive used everywhere.

splitheads(X) = permutedims(reshape(X, size(X, 1), size(X, 2), H, D), (3, 2, 1, 4)) # Turn width C into H heads of width D; which channels form a head is an arbitrary fixed choice.
joinheads(X) = reshape(permutedims(X, (3, 2, 1, 4)), size(X, 3), size(X, 2), C)     # Put those heads back into one residual vector.

function bmm(A, B)                                                  # Batched matrix multiply over head and batch dimensions.
    Y = Array{Float32}(undef, size(A, 1), size(A, 2), size(A, 3), size(B, 4))
    @views for h in 1:size(A, 1), b in 1:size(A, 2)                 # Each head in each batch carries its own small matrix multiply.
        Y[h, b, :, :] .= A[h, b, :, :] * B[h, b, :, :]
    end
    return Y                                                        # Attention is built entirely from these batched multiplies.
end                                                                 # A tiny model can afford this direct implementation.

function softmax_last(x)                                            # One softmax serves logits, attention scores, and sampling.
    d = ndims(x)                                                    # The last dimension always holds the classes or key positions.
    y = x .- maximum(x, dims=d)                                     # Stabilize before exponentiating.
    ex = exp.(y)                                                    # Softmax is just exp followed by normalization.
    return ex ./ sum(ex, dims=d)                                    # Normalize along the last axis only.
end                                                                 # This single definition handles vectors and 4D tensors alike.

function embed(E, tok)                                              # Look up embedding vectors for token ids.
    X = E[vec(tok), :]                                              # Gather one row of E per token in the batch.
    return reshape(X, size(tok, 1), size(tok, 2), size(E, 2))       # Restore time and batch dimensions around the channel width.
end                                                                 # This turns integer tokens into model vectors.

function rms_fwd(X, g)                                              # RMSNorm rescales each token vector by its root-mean-square size.
    r = sqrt.(mean(X .^ 2, dims=3) .+ EPS)                          # Compute one magnitude per token position and batch element.
    Xhat = X ./ r                                                   # Divide by that magnitude to stabilize the residual stream.
    Y = Xhat .* reshape(g, 1, 1, :)                                 # Reintroduce learned per-channel scaling after normalization.
    return Y, (X, Xhat, g, r)                                       # Save the local state required for the backward pass.
end                                                                 # This matches the bias-free RMSNorm variant requested.

function attn_fwd(X, b)                                             # Self-attention lets each position read from its causal past.
    Q = splitheads(linear(X, b.Wq))                                 # Queries ask what each position wants.
    K = splitheads(linear(X, b.Wk))                                 # Keys say what each past position offers.
    Vh = splitheads(linear(X, b.Wv))                                # Values carry the content that may be copied forward.
    scores = bmm(Q, permutedims(K, (1, 2, 4, 3))) ./ sqrt(Float32(D))     # Scale dot products so logits stay in a reasonable range.
    scores .+= reshape(MASK[1:size(X, 1), 1:size(X, 1)], 1, 1, size(X, 1), size(X, 1)) # Adding -1e9 above the diagonal makes exp vanish: zero weight on the future.
    P = softmax_last(scores)                                        # Turn scores into attention weights over visible history.
    A = bmm(P, Vh)                                                  # Average value vectors according to those weights.
    Y = linear(joinheads(A), b.Wo)                                  # Merge heads and project back into residual space.
    return Y, (X, Q, K, Vh, P, A)                                   # Save everything needed for backprop through attention.
end                                                                 # This is the full forward pass of causal multi-head attention.

function block_fwd(X, b)                                            # A block is attention plus an MLP, both wrapped in residual paths.
    X1, c1 = rms_fwd(X, b.g1)                                       # Pre-norm keeps the residual stream stable.
    A, ca = attn_fwd(X1, b)                                         # Read from the causal past.
    H1 = X .+ A                                                     # The update is added, never substituted: blocks learn corrections.
    X2, c2 = rms_fwd(H1, b.g2)                                      # Normalize again before the feed-forward branch.
    Z = linear(X2, b.W1)                                            # Expand the feature space.
    M = max.(0.0f0, Z)                                              # ReLU is the requested MLP nonlinearity.
    H2 = H1 .+ linear(M, b.W2)                                      # Project back down and add another residual update.
    return H2, (M, X2, c1, c2, ca, Z)                               # Save exactly what the backward pass will need.
end                                                                 # Stacking layers just means repeating this same block.

function forward(tok, p)                                            # One forward pass: embeddings -> blocks -> final norm -> tied logits.
    Xtok = embed(p.E, tok)                                          # Convert integer tokens into embedding vectors.
    Xpos = reshape(p.P[1:size(tok, 1), :], size(tok, 1), 1, C)      # Broadcast the matching position embeddings across the batch.
    X = Xtok .+ Xpos                                                # Content and position enter the residual stream together.
    caches = Vector{Any}(undef, length(p.blocks))                   # Each layer will add one entry to the tape.
    for i in eachindex(p.blocks)                                    # Run each Transformer block in sequence.
        X, caches[i] = block_fwd(X, p.blocks[i])                    # The stream is refined; the block's local state is remembered.
    end
    Xf, cf = rms_fwd(X, p.gf)                                       # Normalize once more before scoring the vocabulary.
    logits = linear(Xf, p.E')                                       # The tied head: embedding rows score the very tokens they embed.
    return logits, (caches=caches, Xf=Xf, cf=cf)                    # Logits for predicting, the tape for learning.
end                                                                 # Training and sampling share this single definition of the model.

# =============================================================================
# 3. Learning: Loss, Backward Pass, And Adam
# =============================================================================
#
# Now predictions become a learning signal. Cross-entropy measures the model's
# surprise at the true next character, and its gradient at the logits could not
# be simpler: probs minus onehot. From there the tape is replayed in reverse --
# every operation the forward pass ran now runs backward, handing blame to its
# inputs and its weights -- until Adam turns those gradients into corrections.

function dlinear(dY, X, W)                                          # Backprop through Y = XW.
    dY2 = reshape(dY, :, size(dY, 3))                               # Flatten the output gradient in the same way as the forward pass.
    X2 = reshape(X, :, size(X, 3))                                  # Flatten the input activations so matrix calculus stays visible.
    dX = reshape(dY2 * W', size(X))                                 # Input gradients flow through W transpose.
    dW = X2' * dY2                                                  # Weight gradients are input-transpose times output-gradient.
    return dX, dW                                                   # Linear layers are now explicit in both directions.
end                                                                 # This keeps the code close to the underlying matrix calculus.

function rms_bwd(dY, cache)                                         # Differentiate the RMSNorm step explicitly.
    X, Xhat, g, r = cache                                           # Recover the forward intermediates.
    dXhat = dY .* reshape(g, 1, 1, :)                               # The gain multiplies the upstream gradient channelwise.
    corr = mean(dXhat .* X, dims=3)                                 # The shared denominator couples the channels through one correction term.
    dX = dXhat ./ r .- X .* corr ./ (r .^ 3)                        # This is the derivative of X / rms(X).
    dg = vec(sum(dY .* Xhat, dims=(1, 2)))                          # Gain gradients sum over all time steps and all examples.
    return dX, dg                                                   # Send gradient to the input stream and the gain vector.
end                                                                 # RMSNorm is now fully transparent.

function attn_bwd(dY, cache, b)                                     # Trace the learning signal backward through attention.
    X, Q, K, Vh, P, A = cache                                       # Recover the local forward-pass tensors.
    Acat = joinheads(A)                                             # Recreate the concatenated head representation before Wo.
    dAcat, dWo = dlinear(dY, Acat, b.Wo)                            # Undo the output projection and learn how Wo should change.
    dA = splitheads(dAcat)                                          # Return to per-head layout.
    dP = bmm(dA, permutedims(Vh, (1, 2, 4, 3)))                     # A = P V, so one gradient goes to the attention weights P.
    dVh = bmm(permutedims(P, (1, 2, 4, 3)), dA)                     # The other gradient goes to the value vectors V.
    dS = P .* (dP .- sum(dP .* P, dims=4))                          # Softmax backward without constructing the full Jacobian.
    dS ./= sqrt(Float32(D))                                         # Undo the forward scaling of the attention scores.
    dQ = bmm(dS, K)                                                 # Scores depend linearly on Q when K is fixed.
    dK = bmm(permutedims(dS, (1, 2, 4, 3)), Q)                      # Scores also depend linearly on K when Q is fixed.
    dQcat = joinheads(dQ)                                           # Move query gradients back into residual width C.
    dKcat = joinheads(dK)                                           # Move key gradients back into residual width C.
    dVcat = joinheads(dVh)                                          # Move value gradients back into residual width C.
    dXq, dWq = dlinear(dQcat, X, b.Wq)                              # Learn how the query projection should change.
    dXk, dWk = dlinear(dKcat, X, b.Wk)                              # Learn how the key projection should change.
    dXv, dWv = dlinear(dVcat, X, b.Wv)                              # Learn how the value projection should change.
    dX = dXq .+ dXk .+ dXv                                          # All three branches feed back into the same residual stream.
    return dX, (Wq=dWq, Wk=dWk, Wv=dWv, Wo=dWo)                     # Package the attention gradients for the caller.
end                                                                 # Attention now learns by explicit matrix calculus.

function block_bwd(dH2, cache, b)                                   # Reverse the block in the opposite order from the forward pass.
    M, X2, c1, c2, ca, Z = cache                                    # Recover the saved block-local state.
    dM, dW2 = dlinear(dH2, M, b.W2)                                 # Differentiate the second MLP projection.
    dZ = dM .* (Z .> 0.0f0)                                         # ReLU passes gradient only where its input was positive.
    dX2, dW1 = dlinear(dZ, X2, b.W1)                                # Differentiate the first MLP projection.
    dH1_norm, dg2 = rms_bwd(dX2, c2)                                # Push the MLP gradient through the second RMSNorm.
    dH1 = dH2 .+ dH1_norm                                           # Add the residual shortcut from H2 = H1 + ...
    dX1, datt = attn_bwd(dH1, ca, b)                                # Backpropagate through attention.
    dX_norm, dg1 = rms_bwd(dX1, c1)                                 # Then through the first RMSNorm.
    dX = dH1 .+ dX_norm                                             # Add the first residual shortcut from H1 = X + ...
    grads = BlockParams(dg1, datt.Wq, datt.Wk, datt.Wv, datt.Wo, dg2, dW1, dW2)
    return dX, grads                                                # Return the upstream gradient and all block-local parameter gradients.
end                                                                 # The whole block now learns by explicit chain rule.

function embed_scatter(tok, dX, V)                                  # Accumulate input-embedding gradients back into E.
    dE = zeros(Float32, V, size(dX, 3))                             # Only rows that were actually used receive gradient mass.
    @views for b in 1:size(tok, 2), t in 1:size(tok, 1)
        dE[tok[t, b], :] .+= dX[t, b, :]
    end
    return dE                                                       # This is the reverse of embedding lookup.
end                                                                 # Tied embeddings later add output-side gradients too.

function loss_and_grad(tok, nxt, p)                                 # How wrong is the model, and how should every weight move?
    logits, tape = forward(tok, p)                                  # Think forward once, keeping the tape.
    probs = softmax_last(logits)                                    # Convert logits into next-token distributions.
    true_probs = Matrix{Float32}(undef, size(nxt))                  # Cross-entropy only needs the probability of the true next token.
    @inbounds for b in 1:size(nxt, 2), t in 1:size(nxt, 1)
        true_probs[t, b] = probs[t, b, nxt[t, b]]                   # Gather that probability directly instead of building one-hot targets.
    end
    L = -mean(log.(true_probs))                                     # Average surprise: -log p is small only when the truth was expected.
    dlogits = copy(probs)                                           # The famous shortcut: softmax plus cross-entropy differentiates to...
    @inbounds for b in 1:size(nxt, 2), t in 1:size(nxt, 1)
        dlogits[t, b, nxt[t, b]] -= 1.0f0                           # ...simply probs minus onehot, applied in place.
    end
    dlogits ./= length(nxt)                                         # Average gradients across the whole minibatch.
    dXf, dEt = dlinear(dlogits, tape.Xf, p.E')                      # The tied output head sends gradient to Xf and to E itself.
    dE = permutedims(dEt)                                           # Convert gradient of E' back into gradient of E.
    dX, dgf = rms_bwd(dXf, tape.cf)                                 # Push the loss through the final normalization.
    block_grads = Vector{BlockParams}(undef, length(p.blocks))      # Each layer gets its own gradient bundle.
    for i in length(p.blocks):-1:1
        dX, block_grads[i] = block_bwd(dX, tape.caches[i], p.blocks[i]) # Replay the stack in reverse, from top layer back to bottom.
    end
    dE .+= embed_scatter(tok, dX, size(p.E, 1))                     # What reaches the input stream teaches the tied embeddings too.
    dP = dropdims(sum(dX, dims=2), dims=2)                          # Position embeddings are shared across the batch, so sum their usage.
    G = Params(dE, dP, block_grads, dgf)
    return L, G                                                     # Return one scalar loss and one full gradient bundle.
end                                                                 # This is the entire manual training calculus in one function.

function adam_update!(pk, gk, mk, vk, step)                          # The single Adam recipe every parameter tensor follows.
    @. mk = B1 * mk + (1 - B1) * gk                                  # Update the running gradient average.
    @. vk = B2 * vk + (1 - B2) * (gk * gk)                           # Update the running squared-gradient average.
    @. pk = pk - LR * (mk / (1 - B1^step)) / (sqrt(vk / (1 - B2^step)) + EPS) # Repair the zero-start bias, then step at the gradient's typical scale.
end                                                                  # No scheduler or weight decay is needed for this tiny run.

function adam!(b::BlockParams, g::BlockParams, m::BlockParams, v::BlockParams, step) # Update every tensor inside one Transformer block.
    for name in BLOCK_NAMES                                          # Every block-local tensor follows the same recipe.
        adam_update!(getfield(b, name), getfield(g, name), getfield(m, name), getfield(v, name), step)
    end
end                                                                  # Stacked layers add parameters, never new update rules.

function adam!(p::Params, g::Params, m::Params, v::Params, step)     # Adam updates embeddings, blocks, and the final norm in the same style.
    adam_update!(p.E, g.E, m.E, v.E, step)                           # Token embeddings learn by the shared recipe.
    adam_update!(p.P, g.P, m.P, v.P, step)                           # Position embeddings too.
    for i in eachindex(p.blocks)                                     # Every Transformer layer gets its own Adam step.
        adam!(p.blocks[i], g.blocks[i], m.blocks[i], v.blocks[i], step)
    end
    adam_update!(p.gf, g.gf, m.gf, v.gf, step)                       # The final RMSNorm gain learns like any other parameter.
end                                                                  # Multi-layer Adam is still the same local recipe everywhere.

# =============================================================================
# 4. Training And Inference
# =============================================================================
#
# Here the whole algorithm runs as a process in time. Training repeatedly samples
# short contexts, measures the model's mistake, and updates the weights; once
# learning is done, the same forward pass is reused to speak one token at a time.

function batch(ids, rng)                                            # Build one minibatch of random next-token prediction problems.
    starts = rand(rng, 1:length(ids)-T-1, BATCH)                    # Choose BATCH random start points in the long text stream.
    tok = Matrix{Int}(undef, T, BATCH)                              # Inputs are windows of T characters.
    nxt = Matrix{Int}(undef, T, BATCH)                              # Targets are the same windows shifted by one: every position is an example.
    for (j, i) in enumerate(starts)
        tok[:, j] = ids[i:i+T-1]
        nxt[:, j] = ids[i+1:i+T]
    end
    return tok, nxt                                                 # Character-level GPT needs no more elaborate dataloader than this.
end                                                                 # The corpus is already one long autoregressive stream.

function main()                                                      # Train the model, then sample a few names from it.
    rng = MersenneTwister(SEED)                                      # Fix the random seed so runs are repeatable.
    ids, stoi, itos = load_data()                                    # Load and tokenize the raw character stream.
    p = init_params(rng, length(itos))                               # Initialize the tiny GPT weights.
    m = zeros_like(p)                                                # Adam first moments start at zero.
    v = zeros_like(p)                                                # Adam second moments start at zero.
    for step in 1:STEPS                                              # Repeat batch, loss, gradient, update.
        tok, nxt = batch(ids, rng)

        # --- Form The Loss And Its Gradients -------------------------------
        L, G = loss_and_grad(tok, nxt, p)

        # --- Let Adam Apply The Correction -------------------------------
        adam!(p, G, m, v, step)

        if step % LOG_EVERY == 0
            println("step=", step, " loss=", round(L, digits=4))     # Report training progress occasionally.
        end
    end

    # --- Speak: Inference --------------------------------------------------

    for _ in 1:NUM_SAMPLES                                           # After training, sample several names for inspection.
        out = [stoi[S]]                                              # Begin at the boundary: after a newline comes the start of a name.
        while length(out) < MAX_NEW + 1                              # Grow until the name fills one context window.
            tail = out[max(1, end - T + 1):end]                      # Keep only the visible context window.
            ctx = reshape(tail, :, 1)                                # Turn the prefix into a single-example token matrix.
            logits, _ = forward(ctx, p)                              # Think forward; the tape is not needed just to speak.
            q = softmax_last(vec(logits[end, 1, :]) ./ TEMP)         # Only the last position predicts; TEMP < 1 sharpens its choices.
            u = rand(rng, Float32)                                   # Sample one uniform number in [0,1).
            s = 0.0f0                                                # Accumulate probability mass until we cross the sample.
            nxt = lastindex(q)                                       # Keep the last token as a safe fallback for roundoff.
            @inbounds for i in eachindex(q)
                s += q[i]
                if u <= s
                    nxt = i                                          # The first bin whose mass covers u becomes the sampled token.
                    break
                end
            end
            push!(out, nxt)
            if itos[nxt] == S
                break                                                # A boundary token ends the current sampled name.
            end
        end
        itos[out[end]] == S && pop!(out)                             # Drop the closing boundary when sampling ended at one.
        println(String(itos[out[2:end]]))                            # Drop the opening boundary and print the generated name.
    end
end                                                                  # This closes the full training-and-sampling story.

main()                                                               # Run when invoked as a script.
