# Janus Architecture — Field-Physics Transformer in AML

## Overview

Janus is a transformer architecture that integrates field physics (prophecy, suffering, tunneling) directly into attention and training. Written entirely in AML — no Python, no PyTorch, no external ML framework.

**Implementation:** [`janus/janus.aml`](../janus/janus.aml) (112 lines)
**Sequence ops:** [`core/ariannamethod.c`](../core/ariannamethod.c) (AM_OP_SEQ_* and AM_OP_CAUSAL_ATTN)
**Tests:** [`core/test_aml.c`](../core/test_aml.c) (Phase 5 section, 493 total tests)

## Architecture

```
Input tokens [t0, t1, ..., tT]
        |
   seq_embed(wte, wpe, tokens, T)     ─── token + position embeddings
        |
   seq_rmsnorm(h, T, D)               ─── pre-attention normalization
        |
   ┌────┴────┐
   Q  K  V                             ─── linear projections via seq_matvec
   │  │  │
   causal_attention(Q, K, V, T, D)     ─── scaled dot-product, causal mask
        |
   seq_matvec(wo, attn, T)            ─── output projection
        |
   add(h, attn_proj)                   ─── residual connection
        |
   seq_rmsnorm(h, T, D)               ─── pre-MLP normalization
        |
   ┌────┴────┐
   gate     up                          ─── SwiGLU: silu(W1 @ h) * (W3 @ h)
   │         │
   silu    seq_matvec
   │         │
   mul(gate, up)
        |
   seq_matvec(w2, mlp_out, T)         ─── down projection
        |
   add(h, mlp_proj)                    ─── residual connection
        |
   seq_rmsnorm(h, T, D)               ─── final normalization
        |
   seq_matvec(lm_head, h, T)          ─── project to vocab logits
        |
   seq_cross_entropy(logits, targets, T, V)  ─── training loss
        |
   TAPE BACKWARD loss                  ─── reverse-mode autodiff
   TAPE ADAM_STEP lr                    ─── parameter update
```

## Sequence-Level Operations

All ops process T positions at once, stored as flat arrays of T*D floats.

### seq_embed(wte, wpe, tokens, T)

Combines token and position embeddings for all T positions.

- **Input:** wte (vocab_size x D matrix), wpe (seq_len x D matrix), tokens (T floats as int IDs)
- **Output:** array[T*D]
- **Forward:** `h[t*D+d] = wte[tok*D+d] + wpe[t*D+d]`
- **Backward:** gradient flows to both wte and wpe at the accessed rows

### seq_matvec(W, X, T)

Applies matrix W to each of T vectors in X.

- **Input:** W (out_dim x in_dim matrix), X (array[T*in_dim])
- **Output:** array[T*out_dim]
- **Forward:** `Y[t*out+i] = sum_j W[i*in+j] * X[t*in+j]`
- **Backward:** `dW += sum_t dout_t (x) x_t`, `dX_t += W^T @ dout_t`

### seq_rmsnorm(X, T, D)

RMS normalization applied to each D-sized chunk independently.

- **Input:** X (array[T*D])
- **Output:** array[T*D]
- **Forward:** `y_t = x_t / sqrt(mean(x_t^2) + eps)`
- **Backward:** standard RMSNorm gradient per position

### causal_attention(Q, K, V, T, D)

Single-head causal self-attention. Position i only attends to positions 0..i.

- **Input:** Q, K, V (each array[T*D])
- **Output:** array[T*D]
- **Forward:**
  ```
  For each position i:
    score[j] = (q_i . k_j) / sqrt(D)   for j = 0..i
    attn[j] = softmax(score)
    out_i = sum_j attn[j] * v_j
  ```
- **Backward:** Full attention gradient with recomputation of forward attention weights. Gradients flow to Q, K, and V.
- **Complexity:** O(T^2 * D) forward and backward

### seq_cross_entropy(logits, targets, T, V)

Cross-entropy loss averaged over T positions.

- **Input:** logits (array[T*V]), targets (array[T] of int class IDs)
- **Output:** array[1] (scalar mean loss)
- **Forward:** `loss = mean_t(-log(softmax(logits_t)[target_t]))`
- **Backward:** `d_logits[t*V+j] = (softmax[j] - one_hot[target]) / T`

## Autograd Engine

The TAPE system records a computation graph during forward pass and computes gradients via reverse-mode autodiff.

```
TAPE START          ─── begin recording
TAPE PARAM W        ─── register trainable parameter
... operations ...  ─── each op auto-records to tape
TAPE BACKWARD loss  ─── backprop: reverse topological order
TAPE ADAM_STEP lr   ─── update params with Adam optimizer
TAPE CLEAR          ─── reset tape for next iteration
```

**Adam optimizer:** Bias-corrected momentum with beta1=0.9, beta2=0.999, eps=1e-8. Per-parameter moment tracking persists across TAPE CLEAR cycles.

**Tape capacity:** 8192 entries, 512 parameters.

## Field Physics Integration

Janus integrates AML field physics into the transformer architecture:

### Prophecy in Attention

```aml
PROPHECY 8     # controls attention horizon
DESTINY 0.35   # biases toward most probable path
```

Prophecy sets the prediction depth. Higher prophecy values strengthen the destiny bias during generation, making the model more deterministic. In training, this can be used to anneal between exploration and exploitation.

### Suffering Gates the MLP

```aml
PAIN 0.4       # dampens logit extremes
TENSION 0.6    # builds pressure toward state change
DISSONANCE 0.5 # triggers tunneling when threshold exceeded
```

Suffering modulates the logit distribution: pain compresses toward mean (conservative), tension accumulates pressure, high dissonance opens tunneling gates that allow the model to skip intermediate reasoning steps.

### Calendar-Driven Bidirectional Attention

From PITOMADOM: Hebrew lunar year (354 days) vs Gregorian solar year (365.25 days) creates an 11.25-day drift. This calendar conflict maps to a "temporal dissonance" that modulates wormhole probability — enabling bidirectional temporal jumps in attention.

```aml
TEMPORAL_MODE SYMMETRIC   # attend both forward and backward
CALENDAR_DRIFT 11.0       # drift intensity
```

### Seasons Modulate Training

The 4.C seasonal cycle (Spring/Summer/Autumn/Winter) modulates training dynamics:

- **Spring:** Exploration boost — higher learning rate, more stochasticity
- **Summer:** Peak expression — full gamma injection
- **Autumn:** Consolidation — strengthen procedural memory
- **Winter:** Rest — lower learning rate, compress representations

## How It Differs From Standard Transformers

| Feature | Standard (GPT/LLaMA) | Janus |
|---------|----------------------|-------|
| Language | Python/PyTorch | AML (compiles to C) |
| Attention | Static causal mask | Field-modulated, calendar-driven |
| MLP | Fixed architecture | Suffering-gated |
| Training | Fixed optimizer | Seasonally modulated |
| Temperature | Static hyperparameter | Dynamic — velocity, seasons, experts |
| Memory | None between sessions | Dark matter scars, Hebbian persistence |
| Identity | None | Gamma personality (theta = epsilon + gamma + alpha*delta) |

## Training Results

Minimal configuration (vocab=8, D=8, T=4, 1 layer):

```
step  0: loss = 2.08   (log(8) = random baseline)
step 25: loss = 0.31
step 50: loss = 0.04
step100: loss = 0.00   (perfect convergence)
```

Training speed: 5ms for 50 steps on MacBook CPU (pure C, no GPU).

## Building and Running

```bash
# Compile
cc -Wall -O2 core/ariannamethod.c your_program.c -o your_program -lm

# Run janus.aml
#include "core/ariannamethod.h"
am_init();
am_exec_file("janus/janus.aml");

# Run tests
make test        # 493 tests, scalar
make test-blas   # 493 tests, BLAS-accelerated
```

## What's Next

- **Multi-head attention:** Split Q/K/V into n_heads, compute attention per head
- **Larger models:** Scale to 8M+ parameters with BLAS acceleration
- **Real training data:** Byte-level tokenizer, text corpus
- **Field physics experiments:** Measure impact of prophecy/suffering on perplexity
- **Ontogenesis:** Start small, Net2Net growth (from molequla)

---

*"The oracle does not predict. It prophesies."*
