# Train-Time LoRA Variant: SP8192 + 3-Layer Recurrence + Parallel Residuals + QK-Gain 5.25 + Legal TTT

This directory documents an experimental train-time LoRA variant of the
`2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT` record.

No run result is claimed here. The purpose of this package is to preserve the
implementation details for using low-rank trainable adapters while reconstructing
the frozen base model from initialization during serialization.

## Key Techniques

1. **Inherited SP8192 Base** - starts from the 2026-04-09 architecture: SP8192 tokenizer, depth recurrence, parallel residuals, QK gain, tied embeddings, GPTQ, and legal score-first TTT.
2. **Train-Time LoRA** - optional `LORA_RANK` inserts LoRA adapters around `CastedLinear` modules. Base linear weights are frozen, while LoRA adapters and non-linear/control tensors remain trainable.
3. **Frozen-A Mode** - optional `LORA_FREEZE_A=1` freezes the randomly initialized LoRA A matrices so they can be reconstructed from the initialization seed instead of stored.
4. **Compact Artifact Mode** - LoRA artifacts store only non-reconstructible tensors. Frozen base weights, biases, and optionally frozen LoRA A tensors are rebuilt from the seed at deserialize time.
5. **Fused Eval Weights** - deserialization rebuilds the LoRA model, fuses each low-rank delta into its base weight, and loads the fused state into a plain non-LoRA model for validation and TTT.
6. **LoRA-Aware GPTQ** - LoRA tensors use a separate smaller GPTQ threshold so large adapter matrices can be quantized instead of being forced through the regular full-model threshold.

## Architecture

The inherited base is 11L x 512d x 8H / 4KV with MLP 4x, LeakyReLU(0.5)^2,
Partial RoPE, layerwise LN scale, tied embeddings, logit softcap=30.0, depth
recurrence over layers 3-5, parallel residuals from the decoder tail, and
sigmoid-gated U-Net skip connections.

LoRA is applied after model initialization, so setting `LORA_RANK=0` should
recover the inherited full-model training path. Setting `LORA_RANK>0` replaces
eligible casted linear layers with LoRA-wrapped modules.

## Training

The inherited optimizer split is preserved: AdamW for token embeddings and scalar
parameters, Adam for the LM head when untied, and Muon for trainable matrix
parameters. Frozen base weights are skipped by the optimizer because their
`requires_grad` flag is disabled.

Useful knobs:

```bash
LORA_RANK=384
LORA_FREEZE_A=1
MODEL_DIM=768
EMBEDDING_DIM=768
MLP_MULT=4.3333333333
QK_GAIN_INIT=5.25
TTT_ENABLED=1
TTT_LR=0.005
TTT_EPOCHS=3
```

`MAX_TRAIN_STEPS=N` optionally stops the main training loop after `N` optimizer
steps. It combines with `MAX_WALLCLOCK_SECONDS`, and training exits when either
cap is reached. Use `MAX_WALLCLOCK_SECONDS=0 MAX_TRAIN_STEPS=N` for a
step-only cap.

## Serialization

When LoRA is disabled, serialization follows the inherited full-state path.

When LoRA is enabled, the artifact metadata records `artifact_mode`, `init_seed`,
`lora_rank`, and `lora_freeze_a`. Deserialization uses those fields to rebuild
the matching initialized model, load the compact adapter state, fuse LoRA deltas
into regular weights, and evaluate with a plain model state dict.

## GPTQ

The compact state is quantized with the existing GPTQ pipeline. LoRA tensors use
`LORA_GPTQ_MIN_NUMEL`, while inherited full-model tensors use the standard
`GPTQ_MIN_NUMEL`. Hessian hooks collect the correct inputs for LoRA A and LoRA B
when those adapter tensors meet the quantization threshold.

## TTT Compliance

The inherited legal score-first TTT path is unchanged. Evaluation scores tokens
before any eval-time update, uses a normalized full-vocabulary softmax, and does
not add n-gram caches, logit biasing, or rescoring.

## Reproduction Skeleton

```bash
pip install brotli sentencepiece
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192

SEED=42 \
LORA_RANK=384 \
LORA_FREEZE_A=1 \
MODEL_DIM=768 \
EMBEDDING_DIM=768 \
MLP_MULT=4.3333333333 \
QK_GAIN_INIT=5.25 \
TTT_ENABLED=1 \
TTT_LR=0.005 \
TTT_EPOCHS=3 \
torchrun --standalone --nproc_per_node=8 train_gpt_decoded.py
```

## Credits

- **@clarkkev** - SP8192, GPTQ embeddings, SDClip, MuonEq-R, and depth-recurrence lineage.
- **@dexhunter** - 3-layer depth recurrence and legal TTT on SP8192.
- **@abaybektursun** - score-first TTT framework.
- **@Robby955** and **@msisovic** - parallel residual technique lineage.
- **@X-Abhishek-X** - base hyperparameter tuning lineage.

## Included Files

- `README.md` - this documentation file.
- `train_gpt_decoded.py` - train-time LoRA implementation on the 2026-04-09 base.
- `estimate_lora_param_savings.py` - helper for estimating compact LoRA state size.
