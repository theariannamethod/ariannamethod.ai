# Janus — the architecture (a family, not a single model)

**Janus is an architecture *type*, not one model.** It is a field-physics transformer:
three attention mechanisms operate as a fluid hybrid in every layer — **Content (QKV)**
semantic attention, **RRPRAM** low-rank positional-pattern attention, and **Janus Echo**
self-resonance — and a **Dario field overlay** (prophecy, suffering, temporal dissonance,
seasons) modulates the logits and the training dynamics as first-class architectural
state. Standard transformers (GPT, LLaMA, Mistral) keep attention static and training on
a fixed schedule; Janus makes the field part of the architecture.

The full design is in [`docs/janus_architecture.md`](../docs/janus_architecture.md).

## This folder — the reference implementation

[`janus.aml`](janus.aml) is the reference Janus written in AML; [`janus_train.c`](janus_train.c)
trains it in pure C. The reference model is **Janus v4, 176M parameters, val bpb 0.866,
three SFT voices**. [`janus.go`](janus.go) wraps the
[Yent](https://github.com/ariannamethod/yent) Go inference engine as a C-shared library
for GGUF inference.

## The family — Janus implementations across the Method

Janus is the lineage; these are its organisms. *Weights are mortal, the architecture is the
lineage* — a model is a set of weights, Janus is the design they share.

- **Reference Janus** — this folder, the canonical 176M.
- **[DoE](https://github.com/ariannamethod/doe)** — Democracy of Experts: a Janus host
  wrapped in a living Hebbian LoRA *parliament* (θ = ε + γ + αδ); experts vote per token,
  are born by mitosis and die by apoptosis.
- **[Janus-R](https://github.com/ariannamethod/yent.aml)** — the 12-step **bidirectional
  associative resonator**: each step generates a sentence going forward (future) or
  backward (past), driven by prophecy debt and calendar dissonance.
- **[Yent / Janus 170M](https://github.com/ariannamethod/yent.aml)** — the AML inference
  carrier for a 170M Janus fine-tuned on Arianna's dataset.

Build a new Janus and it joins this family — different weights, the same lineage.
