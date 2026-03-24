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

function load_data()                                                # Read the corpus and build the minimal character tokenizer.
    text = read(Downloads.download(URL), String)                    # Treat the entire file as one long stream of characters.
    vocab = sort(collect(Set(text)))                                # Use exactly the unique characters that appear in the data.
    stoi = Dict{Char, Int}(ch => i for (i, ch) in enumerate(vocab)) # Map chars to integer token ids.
    ids = Int[stoi[ch] for ch in text]                              # Encode the full corpus once for efficient random slicing.
    return ids, stoi, vocab                                         # `vocab[i]` is already the inverse map from id back to char.
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
    return BlockParams(                                             # The scaling matches the intended shapes closely.
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
# and the final projection turns the residual stream into next-character logits.

function linear(X, W)                                               # Apply a weight matrix to the last dimension of a 3D tensor.
    X2 = reshape(X, :, size(X, 3))                                  # Collapse time and batch so one matmul handles every token vector.
    return reshape(X2 * W, size(X, 1), size(X, 2), size(W, 2))
end                                                                 # This is the basic projection primitive used everywhere.

function dlinear(dY, X, W)                                          # Backprop through Y = XW.
    dY2 = reshape(dY, :, size(dY, 3))                               # Flatten the output gradient in the same way as the forward pass.
    X2 = reshape(X, :, size(X, 3))                                  # Flatten the input activations so matrix calculus stays visible.
    dX = reshape(dY2 * W', size(X))                                 # Input gradients flow through W transpose.
    dW = X2' * dY2                                                  # Weight gradients are input-transpose times output-gradient.
    return dX, dW                                                   # Linear layers are now explicit in both directions.
end                                                                 # This keeps the code close to the underlying matrix calculus.

splitheads(X) = permutedims(reshape(X, size(X, 1), size(X, 2), H, D), (3, 2, 1, 4)) # Turn width C into H heads of width D.
joinheads(X) = reshape(permutedims(X, (3, 2, 1, 4)), size(X, 3), size(X, 2), C)     # Put those heads back into one residual vector.

function bmm(A, B)                                                  # Batched matrix multiply over head and batch dimensions.
    Y = Array{Float32}(undef, size(A, 1), size(A, 2), size(A, 3), size(B, 4))
    @views for h in 1:size(A, 1), b in 1:size(A, 2)                 # Each head in each batch carries its own small matrix multiply.
        Y[h, b, :, :] .= A[h, b, :, :] * B[h, b, :, :]
    end
    return Y                                                        # Attention is built entirely from these batched multiplies.
end                                                                 # A tiny model can afford this direct implementation.

function softmax_last(x::AbstractVector)                            # Sampling needs a vector softmax.
    y = x .- maximum(x)                                             # Shift for numerical stability.
    ex = exp.(y)                                                    # Exponentiate the centered logits.
    return ex ./ sum(ex)                                            # Normalize into a probability distribution.
end                                                                 # This is the usual softmax in one dimension.

function softmax_last(x::AbstractArray)                             # Training uses softmax over the last axis of 3D or 4D tensors.
    d = ndims(x)                                                    # The last dimension always holds the classes or key positions.
    y = x .- maximum(x, dims=d)                                     # Stabilize before exponentiating.
    ex = exp.(y)                                                    # Softmax is just exp followed by normalization.
    return ex ./ sum(ex, dims=d)                                    # Normalize along the last axis only.
end                                                                 # This single definition handles logits and attention scores.

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

function rms_bwd(dY, cache)                                         # Differentiate the RMSNorm step explicitly.
    X, Xhat, g, r = cache                                           # Recover the forward intermediates.
    dXhat = dY .* reshape(g, 1, 1, :)                               # The gain multiplies the upstream gradient channelwise.
    corr = mean(dXhat .* X, dims=3)                                 # The shared denominator couples the channels through one correction term.
    dX = dXhat ./ r .- X .* corr ./ (r .^ 3)                        # This is the derivative of X / rms(X).
    dg = vec(sum(dY .* Xhat, dims=(1, 2)))                          # Gain gradients sum over all time steps and all examples.
    return dX, dg                                                   # Send gradient to the input stream and the gain vector.
end                                                                 # RMSNorm is now fully transparent.

function attn_fwd(X, b)                                             # Self-attention lets each position read from its causal past.
    Q = splitheads(linear(X, b.Wq))                                 # Queries say what the current position wants.
    K = splitheads(linear(X, b.Wk))                                 # Keys say what each past position offers.
    Vh = splitheads(linear(X, b.Wv))                                # Values carry the content that may be copied forward.
    scores = bmm(Q, permutedims(K, (1, 2, 4, 3))) ./ sqrt(Float32(D))     # Scale dot products so logits stay in a reasonable range.
    scores .+= reshape(triu(fill(-1.0f9, size(X, 1), size(X, 1)), 1), 1, 1, size(X, 1), size(X, 1)) # Mask future positions before softmax.
    P = softmax_last(scores)                                        # Turn scores into attention weights over visible history.
    A = bmm(P, Vh)                                                  # Average value vectors according to those weights.
    Y = linear(joinheads(A), b.Wo)                                  # Merge heads and project back into residual space.
    return Y, (X, Q, K, Vh, P, A)                                   # Save everything needed for backprop through attention.
end                                                                 # This is the full forward pass of causal multi-head attention.

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

function block_fwd(X, b)                                            # A block is attention plus an MLP, both wrapped in residual paths.
    X1, c1 = rms_fwd(X, b.g1)                                       # Pre-norm keeps the residual stream stable.
    A, ca = attn_fwd(X1, b)                                         # Read from the causal past.
    H1 = X .+ A                                                     # Add the attention update back into the stream.
    X2, c2 = rms_fwd(H1, b.g2)                                      # Normalize again before the feed-forward branch.
    Z = linear(X2, b.W1)                                            # Expand the feature space.
    M = max.(0.0f0, Z)                                              # ReLU is the requested MLP nonlinearity.
    H2 = H1 .+ linear(M, b.W2)                                      # Project back down and add another residual update.
    return H2, (M, X2, c1, c2, ca, Z)                               # Save exactly what the backward pass will need.
end                                                                 # Stacking layers just means repeating this same block.

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

function forward(tok, p)                                            # Inference maps a token context to logits over the next character.
    Xtok = embed(p.E, tok)                                          # Convert integer tokens into embedding vectors.
    Xpos = reshape(p.P[1:size(tok, 1), :], size(tok, 1), 1, C)      # Broadcast the matching position embeddings across the batch.
    H = Xtok .+ Xpos                                                # Add content and position information together.
    for b in p.blocks                                               # Run each Transformer block in sequence.
        H, _ = block_fwd(H, b)
    end
    Xf, _ = rms_fwd(H, p.gf)                                        # Normalize once more before producing logits.
    return linear(Xf, p.E')                                         # Tying output weights to E keeps the model especially small.
end                                                                 # The same forward pass is used for training and sampling.

# =============================================================================
# 3. Learning: Loss, Gradients, And Adam
# =============================================================================
#
# Now predictions become a learning signal. Cross-entropy says how wrong the model
# was, the explicit backward equations carry that mistake through every operation,
# and Adam turns those gradients into small, stable corrections to the weights.

function embed_scatter(tok, dX, V)                                  # Accumulate input-embedding gradients back into E.
    dE = zeros(Float32, V, size(dX, 3))                             # Only rows that were actually used receive gradient mass.
    @views for b in 1:size(tok, 2), t in 1:size(tok, 1)
        dE[tok[t, b], :] .+= dX[t, b, :]
    end
    return dE                                                       # This is the reverse of embedding lookup.
end                                                                 # Tied embeddings later add output-side gradients too.

function loss_and_grad(tok, nxt, p)                                 # Training asks how wrong the model is and how every weight should move.
    Xtok = embed(p.E, tok)                                          # Embed the current input characters.
    Xpos = reshape(p.P[1:size(tok, 1), :], size(tok, 1), 1, C)      # Attach position information for the visible context.
    X0 = Xtok .+ Xpos                                               # Form the initial residual stream.
    H = X0                                                          # This stream will be refined by one block after another.
    caches = Vector{Any}(undef, length(p.blocks))                   # Save one cache per layer for the backward pass.
    for i in eachindex(p.blocks)
        H, caches[i] = block_fwd(H, p.blocks[i])                    # Run the whole stack and remember each block's local state.
    end
    Xf, cf = rms_fwd(H, p.gf)                                       # Run the final RMSNorm and save its cache too.
    logits = linear(Xf, p.E')                                       # Score every vocabulary item at every position.
    probs = softmax_last(logits)                                    # Convert logits into next-token distributions.
    true_probs = Matrix{Float32}(undef, size(nxt))                  # Cross-entropy only needs the probability of the true next token.
    @inbounds for b in 1:size(nxt, 2), t in 1:size(nxt, 1)
        true_probs[t, b] = probs[t, b, nxt[t, b]]                   # Gather that probability directly instead of building one-hot targets.
    end
    L = -mean(log.(true_probs))                                     # Cross-entropy rewards putting mass on the true next character.
    dlogits = copy(probs)                                           # Softmax-cross-entropy starts from the probabilities.
    @inbounds for b in 1:size(nxt, 2), t in 1:size(nxt, 1)
        dlogits[t, b, nxt[t, b]] -= 1.0f0                           # Subtract the one-hot target distribution in place.
    end
    dlogits ./= length(nxt)                                         # Average gradients across the whole minibatch.
    dXf, dEt = dlinear(dlogits, Xf, p.E')                           # The tied output head sends gradient to Xf and to E itself.
    dE = permutedims(dEt)                                           # Convert gradient of E' back into gradient of E.
    dH, dgf = rms_bwd(dXf, cf)                                      # Push the loss through the final normalization.
    block_grads = Vector{BlockParams}(undef, length(p.blocks))      # Each layer gets its own gradient bundle.
    for i in length(p.blocks):-1:1
        dH, block_grads[i] = block_bwd(dH, caches[i], p.blocks[i])   # Reverse the stack from top layer back to bottom layer.
    end
    dX0 = dH                                                        # Whatever remains now points at the input residual stream.
    dE .+= embed_scatter(tok, dX0, size(p.E, 1))                    # Tied embeddings also learn from their input-side use.
    dP = dropdims(sum(dX0, dims=2), dims=2)                         # Position embeddings are shared across the batch, so sum their usage.
    G = Params(dE, dP, block_grads, dgf)
    return L, G                                                     # Return one scalar loss and one full gradient bundle.
end                                                                 # This is the entire manual training calculus in one function.

function batch(ids, rng)                                            # Build one minibatch of random next-token prediction problems.
    starts = rand(rng, 1:length(ids)-T-1, BATCH)                    # Choose BATCH random start points in the long text stream.
    tok = Matrix{Int}(undef, T, BATCH)                              # Inputs are windows of T characters.
    nxt = Matrix{Int}(undef, T, BATCH)                              # Targets are the same windows shifted by one.
    for (j, i) in enumerate(starts)
        tok[:, j] = ids[i:i+T-1]
        nxt[:, j] = ids[i+1:i+T]
    end
    return tok, nxt                                                 # Character-level GPT needs no more elaborate dataloader than this.
end                                                                 # The corpus is already one long autoregressive stream.

function adam!(b::BlockParams, g::BlockParams, m::BlockParams, v::BlockParams, step) # Adam turns raw gradients into stable adaptive updates.
    for name in BLOCK_NAMES                                         # Every block-local tensor follows the same recipe.
        pk = getfield(b, name)
        gk = getfield(g, name)
        mk = getfield(m, name)
        vk = getfield(v, name)
        @. mk = B1 * mk + (1 - B1) * gk                              # Update the running gradient average.
        @. vk = B2 * vk + (1 - B2) * (gk * gk)                       # Update the running squared-gradient average.
        mhat = mk ./ (1 - B1^step)                                   # Bias-correct the first moment.
        vhat = vk ./ (1 - B2^step)                                   # Bias-correct the second moment.
        @. pk = pk - LR * mhat / (sqrt(vhat) + EPS)                  # Take the normalized parameter step in place.
    end
end                                                                  # No scheduler or weight decay is needed for this tiny run.

function adam!(p::Params, g::Params, m::Params, v::Params, step)     # Adam updates embeddings, blocks, and the final norm in the same style.
    @. m.E = B1 * m.E + (1 - B1) * g.E                               # Update the running average for token embeddings.
    @. v.E = B2 * v.E + (1 - B2) * (g.E * g.E)                       # Update the running squared average for token embeddings.
    @. p.E = p.E - LR * (m.E / (1 - B1^step)) / (sqrt(v.E / (1 - B2^step)) + EPS) # Take the token-embedding step.
    @. m.P = B1 * m.P + (1 - B1) * g.P                               # Do the same for position embeddings.
    @. v.P = B2 * v.P + (1 - B2) * (g.P * g.P)
    @. p.P = p.P - LR * (m.P / (1 - B1^step)) / (sqrt(v.P / (1 - B2^step)) + EPS)
    for i in eachindex(p.blocks)                                     # Every Transformer layer gets its own Adam step.
        adam!(p.blocks[i], g.blocks[i], m.blocks[i], v.blocks[i], step)
    end
    @. m.gf = B1 * m.gf + (1 - B1) * g.gf                            # Final RMSNorm gain learns like any other parameter.
    @. v.gf = B2 * v.gf + (1 - B2) * (g.gf * g.gf)
    @. p.gf = p.gf - LR * (m.gf / (1 - B1^step)) / (sqrt(v.gf / (1 - B2^step)) + EPS)
end                                                                  # Multi-layer Adam is still the same local recipe everywhere.

# =============================================================================
# 4. Training And Inference
# =============================================================================
#
# Here the whole algorithm runs as a process in time. Training repeatedly samples
# short contexts, measures the model's mistake, and updates the weights; once
# learning is done, the same forward pass is reused to speak one token at a time.

function main()                                                      # Train the model, then sample a few names from it.
    rng = MersenneTwister(SEED)                                      # Fix the random seed so runs are repeatable.
    ids, stoi, itos = load_data()                                    # Load and tokenize the raw character stream.
    p = init_params(rng, length(itos))                               # Initialize the tiny GPT weights.
    m = zeros_like(p)                                                # Adam first moments start at zero.
    v = zeros_like(p)                                                # Adam second moments start at zero.
    for step in 1:STEPS                                              # Repeat batch, loss, gradient, update.
        tok, nxt = batch(ids, rng)

        # --- Form The Loss -------------------------------------------------
        L, G = loss_and_grad(tok, nxt, p)

        # --- Let Adam Apply The Correction -------------------------------
        adam!(p, G, m, v, step)

        if step % LOG_EVERY == 0
            println("step=", step, " loss=", round(L, digits=4))     # Report training progress occasionally.
        end
    end

    # --- Speak: Inference --------------------------------------------------

    for _ in 1:NUM_SAMPLES                                           # After training, sample several names for inspection.
        out = [stoi[S]]                                              # Start from the shared boundary token.
        while length(out) <= MAX_NEW + 1
            tail = out[max(1, end - T + 1):end]                      # Keep only the visible context window.
            ctx = reshape(tail, :, 1)                                # Turn the prefix into a single-example token matrix.
            logits = forward(ctx, p)                                 # Ask the model for logits at every visible position.
            q = softmax_last(vec(logits[end, 1, :]) ./ TEMP)         # Use only the final position and apply temperature.
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
        println(String(itos[out[2:end-1]]))                          # Drop boundary tokens and print the generated name.
    end
end                                                                  # This closes the full training-and-sampling story.

main()                                                               # Run when invoked as a script.
