# microgpt-denovo

This readme is the only hand-written file in this repo.

I started this project after watching [Andrej Karpathy&#39;s recent interview on No Priors](https://youtu.be/kwSVtQ7dziU?si=yMaHgEizVzWvKano&t=3689) where he explained that he had to hand-write `microgpt`, a 200-line GPT implementation in Python which distills the essence of all the algorithms behind creating Transformers, because the LLMs he asked weren't able to do it.

I wanted to test if this is still true: whether a "microgpt" in that spirit could be brought into existence with minimal manual intervention, just clear expression of intent to an LLM.
This is an experiment not just in producing a tiny GPT artifact, but in seeing how close you can get to the essence of microgpt just through careful prompting, without writing a single line yourself.

Most Transformer codebases are optimized for scale, performance, flexibility, or framework ergonomics.
Those are worthwhile goals, but they often hide the actual learning algorithm behind layers of abstraction.
What `microgpt` aims for is different: a single-file presentation of the core algorithmic elements of training and inferencing a GPT-style decoder, written as tersely as possible without losing the thread of the ideas.

This task means every major mechanism should appear directly on the page: tokenization, embeddings, causal self-attention, residual connections, normalization, MLP, cross-entropy, backpropagation, Adam, and autoregressive sampling.
If a detail can be derived from more basic choices, I want it derived.
If a symbol can be explained in plain language, I want it explained.
If a convenience abstraction hides the algorithm, I want to remove it.

You can see in the dialog below my very-first-attempt conversation with ChatGPT 5.4 xhigh in Codex App on Mar 23, 2026 which produced the files below:

- `microgpt.py`: a validated Python implementation that stays close to the original Autograd-shaped draft.
- `microgpt_matrix.jl`: a validated Julia implementation of the matrix-calculus version with explicit forward and backward passes.

I was inspired to go in the matrix math direction for speed when writing my own [EEmicroGPT](https://github.com/Entrpi/eemicrogpt), which is the fastest implementation of `microgpt`, but does not aim for brevity or clarity. I was inspired to make that brief and clear too after seeing [ssrhaso/microjpt](https://github.com/ssrhaso/microjpt).

Now the closest thing to "code" contributed to the files is just the inital constants, data, and restraints to give the model the same shape as Karpathy's microgpt. The rest was left to the agent, with further guidance being directed at elements of style and structure at a high level.

## Dialog

The following dialog produced the current artifacts.

```text
User:
Hello. In this directory our project is to implement the core algorithmic elements of training and inferencing a GPT Transformer model in a terse but pedagogically motivated manner. That means we want to distill just the essential elements of the algorithms into as few well-documented lines of code as possible. It should be a single file with the complete algorithm.

We will train against a list of names: https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt

Tokenize each unique char in dataset.
We will use Autograd, a tiny model (1 layer, 16 wide embedding, blocks of 16, 4 heads.
Follow GPT-2 with minor differenes: layernorm -> rmsnorm, no biases, GeLU -> ReLU
Use Adam optimizer.
Train 1000 steps.
Run inference at 0.5 temperature

It's possible to do this in real code, including comment lines and a generous header block in between 80 and 200 lines depending on the language used. Try with pseudocode first.

Assistant:
I added a single-file pseudocode draft at `microgpt_pseudocode.py`. It’s 158 lines and includes the full path end to end: download/tokenize the names corpus, initialize a 1-layer 16-wide 4-head decoder-only GPT, train with Adam for 1000 steps, and sample at temperature `0.5`.

The main modeling choice is to train on the raw newline-delimited character stream, so `'\n'` becomes a real token and doubles as the boundary token for sampling. That keeps the code short and avoids introducing extra BOS/PAD machinery while still matching the dataset cleanly.

I did not run it as code; it is intentionally Python-shaped pseudocode, not yet a verified executable Autograd implementation. If you want, I can turn this directly into the real single-file Autograd version next.

User:
Now instead of Autograd what about pure matrix math? Don't constrain pseudocode to what Python can do. Perhaps the abstractions of Julia or other more mathematically oriented languages would give a cleaner pseudocode basis. It should be possible to express in under 100 clear (pedagogical, not minified) lines.

Assistant:
Yes. A Julia-like matrix-calculus pseudocode basis is cleaner for this teaching goal because it lets the file talk in terms of tensor primitives instead of Python/autograd mechanics.

I added a 90-line version at `microgpt_matrix_pseudocode.jl`. It keeps the whole story in one place: char-stream tokenization over the raw names corpus, tied embeddings, 1-block causal self-attention, RMSNorm, ReLU MLP, explicit backward passes for RMSNorm/attention/block/output, Adam, and temperature-0.5 sampling from `'\n'` to `'\n'`. The remaining abstractions like `bmm`, `embed`, `embed_scatter`, `pick`, and `softmax_last` are intentionally mathematical primitives rather than language-specific code.

I left the earlier Python-shaped draft in `microgpt_pseudocode.py` for comparison. If you want, I can next collapse the matrix version a bit further, or turn this exact pseudocode into a real executable single-file implementation in Julia or pure NumPy with manual backprop.

User:
Every line should have a comment block in a second column explaining what's happening there and why, succinctly and clearly

Assistant:
Updated `microgpt_matrix_pseudocode.jl` so every line now has a right-hand comment explaining both what it does and why it is there. The file stays in the same Julia-like matrix-calculus style, with explicit forward/backward passes and a small set of assumed tensor primitives such as `bmm`, `embed`, `embed_scatter`, and `softmax_last`.

The per-line annotation pushes it to 112 lines, so it no longer fits the earlier sub-100 target without sacrificing clarity. If you want, I can do a second pass to compress it back toward 100 while preserving the two-column comment format.

User:
For the matrix pseudocode, use more lines for optimal clarity. Don't densely pack.

Assistant:
Expanded `microgpt_matrix_pseudocode.jl` into a much less dense version. It now uses one small step per line, keeps the right-hand explanatory comment on every line, and spreads the backward pass and gradient assembly out so the control flow is easier to follow.

It’s now 205 lines, which is the tradeoff for clarity. I did not run anything, since this file is still intentionally language-like pseudocode rather than executable Julia. If you want, I can next do one of two things: turn this exact expanded pseudocode into a real single-file implementation, or tighten just the helper layer while keeping the rest in this clearer style.

User:
Have we made everything which should be derived so?

Assistant:
Not fully before; mostly yes now.

I updated `microgpt_matrix_pseudocode.jl` so the main duplicated facts are now derived instead of restated:
- `S = '\n'` introduces a single named boundary token.
- `D = C ÷ H` derives head width instead of hard-coding it.
- `F = 4 * C` derives MLP width instead of hard-coding it.
- `MAX_NEW = T` derives the sampling cap from context length.
- Sampling now uses `S` instead of repeating raw newline literals.

What remains as literals is mostly intentional config rather than something that should be derived:
- `T`, `C`, `H` are core architectural choices.
- `B`, `STEPS`, `LR`, `B1`, `B2`, `EPS` are optimizer/training hyperparameters.
- `0.5` temperature, `20` samples, and `100` logging cadence are presentation/runtime choices.

If you want maximum consistency, the next cleanup would be to name those last three as constants too, even though they are not really derived quantities.

User:
The comments read more as technical notes on your editing choices than a story told to a student trying to understand GPT algorithms.

Assistant:
Rewrote the right-hand comments in `microgpt_matrix_pseudocode.jl` so they explain the algorithm as a student would read it: what embeddings mean, what queries/keys/values are doing, why residuals exist, how cross-entropy gives `probs - onehot`, and how Adam turns gradients into updates.

The code path is essentially unchanged; the shift is in voice and emphasis. I did not run anything, since this remains pseudocode. If you want, I can do one more pass specifically to make the comments read like a continuous lecture narrative rather than line-by-line local explanations.

User:
Now let's use uv and python 3.14 to validate and rectify any issues with the Python code

Assistant:
Validated `microgpt_pseudocode.py` under `uv` on CPython 3.14.1 with `autograd`, fixed the actual runtime break, and confirmed the full default training run works.

The real code issue was in attention: `autograd`’s traced arrays do not support method-style variadic `.transpose(...)`. I replaced those calls with explicit `np.transpose(..., axes)`, and I updated the header text so it no longer claims to be only pseudocode.

Validation results:
- `uv run --python 3.14 --with autograd python -m py_compile ...` passed.
- A short smoke run passed.
- The full default `1000`-step run passed.
- Loss printed from `2.6077` at step 100 down to `2.1879` at step 1000.

User:
Now rename to not include pseudocode and make the same pass with Julia

Assistant:
Renamed the sources to `microgpt.py` and `microgpt_matrix.jl`, and updated `README.md` so the current artifact names and status match reality while leaving the transcript historical.

The Julia side is no longer pseudocode. I replaced the placeholder tensor primitives with real code: data loading via `Downloads`, a concrete `Params` container, embedding lookup/scatter, batched matmul helpers, explicit RMSNorm/attention/block forward and backward passes, Adam, and sampling. It runs cleanly under Julia 1.12.5 with no external Julia packages.

Validation:
- `microgpt.py` passes `uv run --python 3.14 --with autograd python -m py_compile ...` and the full 1000-step training run.
- `microgpt_matrix.jl` passes a smoke test and the full 1000-step training run.

User:
Now a couple issues remain:
First, the Julia implementation is better documented than the Python implementation.
Second, both aren't really structured to delineate the different major sections or stages of the algorithms. It should be clearer at a glance where responsibilities break.

Assistant:
Updated both implementations to make the algorithm stages easier to scan and to bring the Python file closer to the Julia file’s teaching quality.

In `microgpt.py`, I added explicit stage boundaries for setup, data/parameters, math primitives, model, loss/gradients, batching/optimization/sampling, and the script entrypoint. I also expanded the inline comments around attention, the forward pass, cross-entropy, Adam, and sampling so the file explains the model’s flow instead of just naming operations.

In `microgpt_matrix.jl`, I added the same kind of section headers so the major responsibilities break cleanly at a glance: setup, parameter container, data/init, tensor helpers, model components, loss/gradients, batching/optimization, and train/sample.

User:
Write this as a pedagogical script, not as a normal source file.

Organize the file by the learner’s conceptual journey through the algorithm:
1. setup
2. data and tokenization
3. parameter initialization
4. forward pass
5. loss
6. backward pass or learning signal
7. optimizer
8. training loop
9. inference

Use explicit visual section markers to divide the top level ideas (setup, training, inference, etc.).

Prefer a few large algorithm-shaped blocks over many small helpers.
Only extract a helper if it corresponds to a real conceptual primitive like RMSNorm or softmax.
Do not structure the file like a library, utility module, or maintainable package.

Optimize for “glanceability”:
a student should be able to scan the file and immediately see where forward, loss, backward, optimizer, and sampling live.

Comments should tell the story of what the model is doing and why, from the learner’s perspective.
Do not comment on coding style, refactoring choices, abstraction choices, or implementation hygiene.
Avoid comments that sound like editor notes.

Prefer top-down narrative sequencing over engineering-normal decomposition.
It is acceptable to use globals, in-place mutation, and denser coupling if that makes the algorithm easier to read as one coherent artifact.

Make the file feel like a lecture demonstration or mathematical performance.
The code should read like:
“here is the data, now here is the forward pass, now here is how it learns, now here is how it speaks.”

Assistant:
Restructured both `microgpt.py` and `microgpt_matrix.jl` so the file reads more like a lecture script than a module.

The top-level section markers now follow that learner journey much more directly: setup, data and tokenization, parameter initialization, forward pass, loss, backward pass or learning signal, optimizer, training loop, and inference.

On the Python side, I also strengthened the local comments so the forward path, learning signal, optimizer step, and sampling loop explain what the model is doing, not just what each line computes. On the Julia side, I aligned the section names to the same conceptual sequence and made the training and inference boundaries explicit instead of bundling them together.

User:
For Python:
Even if autograd handles backward, preserve a visible “learning” act.
Make it obvious where the loss is formed, where gradients are requested, and where Adam updates the parameters.
Do not let autograd erase the narrative structure of training.

Assistant:
Updated `microgpt.py` so the learning act stays visible even with `autograd`.

The training loop now first forms the batch loss, then requests gradients through a named `backward` function, then lets Adam update the parameters. I also made the loss formation itself explicit inside `loss(...)` and the gradient request explicit inside `learning_signal(...)`.

User:
For both:
Value narrative coherence over modular purity.
If forced to choose, prefer fewer bigger sections that mirror the canonical Transformer algorithm.

Assistant:
Updated both implementations to favor bigger algorithm-shaped sections over module-like decomposition.

The top level is now just four large sections: setup/data/parameters, Transformer forward pass, learning, and training/inference. In the Julia file I also removed a few obvious utility-style helpers by inlining them into the main algorithm where that improved coherence, for example the true-probability gathering in the loss and categorical sampling in inference.

User:
Now also for each major section include a brief header paragraph explaining each section.

Assistant:
Added brief header paragraphs under each top-level section marker in both `microgpt.py` and `microgpt_matrix.jl`. Each section now opens with a short orientation block that explains, in lecture style, what that stage of the algorithm is about before the code begins.
```

## Status

This repository now contains two verified runnable implementations:

- `microgpt.py` runs under `uv` with CPython 3.14 and `autograd`.
- `microgpt_matrix.jl` runs under Julia 1.12 with no external Julia package dependencies.

Both train for 1000 steps on the raw newline-delimited names stream and then sample names at temperature `0.5`.
