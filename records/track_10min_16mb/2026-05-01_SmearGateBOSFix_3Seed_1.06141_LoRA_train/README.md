# Train-Time LoRA Variant: SmearGate BOS Fix + 3-Seed SOTA Base

This directory documents an experimental train-time LoRA variant rebased onto
`2026-04-29_SmearGateBOSFix_3Seed_1.06141`.

No run result is claimed here. The purpose of this package is to preserve the
implementation details for training low-rank adapters over the current SmearGate
SOTA code while keeping the existing phased eval-time TTT LoRA path intact.

## Key Techniques

1. **Inherited SmearGate SOTA Base** - starts from the 2026-04-29 3-seed compliance reproduction of PR #1851: SmearGate BOS fix, CaseOps SP8192, LQER asymmetric quantization, sparse attention gate, fused softcapped CE, and phased TTT.
2. **Banked Train-Time LoRA** - optional `LORA_RANK` adds low-rank adapters over the banked model matrices: `qo_bank`, `kv_bank`, `mlp_up_bank`, and `mlp_down_bank`.
3. **Frozen Base Banks** - when train-time LoRA is enabled, the base bank tensors are frozen and only adapter matrices plus the normal scalar/control tensors remain trainable.
4. **Frozen-A Mode** - optional `LORA_FREEZE_A=1` freezes the randomly initialized LoRA A matrices so they can be reconstructed from the initialization seed instead of stored.
5. **Compact Artifact Mode** - LoRA artifacts store adapter B tensors, optionally trainable adapter A tensors, and non-reconstructible non-bank tensors. Base bank weights are rebuilt from seed at deserialize time.
6. **Fused Eval Weights** - deserialization rebuilds the LoRA model, fuses low-rank deltas into the four bank tensors, and loads the fused state into a plain non-LoRA model for post-quant eval and phased TTT.
7. **LoRA-Aware GPTQ** - LoRA tensors use `LORA_GPTQ_MIN_NUMEL`, while inherited full-model tensors use the regular `GPTQ_MIN_NUMEL` threshold.

## Architecture

The inherited base is 11L x 512d x 8H / 4KV with MLP 4x,
LeakyReLU(0.5)^2, Partial RoPE, layerwise LN scale, tied embeddings, logit
softcap=30.0, depth recurrence over layers 3-5, parallel residuals from layer 8,
SmearGate with BOS masking, sparse attention gating, CaseOps SP8192, and LQER
asymmetric quantization.

The model stores attention and MLP projection weights in four bank tensors rather
than per-block `CastedLinear` modules. Train-time LoRA therefore attaches one
banked adapter module per bank and indexes the matching low-rank delta inside
`_bank_weights()`.

Setting `LORA_RANK=0` should recover the inherited full-model path. Setting
`LORA_RANK>0` enables train-time banked LoRA while preserving the existing
`BatchedTTTLoRA` eval-time adaptation code.

## Training

The inherited optimizer split is preserved: AdamW for token embeddings and
scalar/control parameters, and Muon for trainable matrix parameters. With
train-time LoRA enabled, frozen base banks are skipped and trainable LoRA
parameters are added to the Muon matrix group.

Useful knobs:

```bash
LORA_RANK=384
LORA_FREEZE_A=1
CASEOPS_ENABLED=1
EMBED_BITS=7
SMEAR_GATE_ENABLED=1
SPARSE_ATTN_GATE_ENABLED=1
MIN_LR=0.1
EMBED_CLIP_SIGMAS=15.0
MLP_CLIP_SIGMAS=12.0
GPTQ_RESERVE_SECONDS=8.0
PHASED_TTT_NUM_PHASES=3
```

## Serialization

When LoRA is disabled, serialization follows the inherited full-state path.

When LoRA is enabled, artifact metadata records `artifact_mode`, `init_seed`,
`lora_rank`, and `lora_freeze_a`. Deserialization uses those fields to rebuild
the matching initialized model, load the compact adapter state, fuse the LoRA
deltas into the bank tensors, and evaluate with a plain model state dict.

## GPTQ

The compact state is unbanked before GPTQ so inherited quantization metadata
still uses familiar flat names for regular bank weights. Train-time LoRA tensors
remain named by adapter module, e.g. `qo_lora.lora_A` and `qo_lora.lora_B`.

Hessian collection records the appropriate inputs for regular bank weights and
for LoRA A/B tensors when those adapter tensors meet the LoRA-specific GPTQ
threshold.

## TTT Compliance

The inherited phased eval-time TTT path is unchanged. Train-time LoRA affects the
trained artifact only; post-quant phased TTT still uses the existing score-first
batched LoRA adapters and document-boundary handling from the SmearGate SOTA
base.

## Reproduction Skeleton

```bash
pip install brotli python-minifier sentencepiece
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
python3 prepare_caseops_data.py

SEED=42 \
LORA_RANK=384 \
LORA_FREEZE_A=1 \
CASEOPS_ENABLED=1 \
EMBED_BITS=7 \
SMEAR_GATE_ENABLED=1 \
SPARSE_ATTN_GATE_ENABLED=1 \
MIN_LR=0.1 \
EMBED_CLIP_SIGMAS=15.0 \
MLP_CLIP_SIGMAS=12.0 \
GPTQ_RESERVE_SECONDS=8.0 \
PHASED_TTT_NUM_PHASES=3 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Helper

`estimate_lora_param_savings.py` instantiates the local model and compares full
state dict size with compact LoRA state size for one or more `MODEL_DIM:LORA_RANK`
configs.

Example:

```bash
python estimate_lora_param_savings.py --config 512:384 --lora-freeze-a
```

## Credits

- **@aquariouseworkman** - PR #1851 SmearGate BOS fix and original seed 42 result.
- **@Christopher-Lee-McClendon** - 3-seed compliance reproduction and GPTQ reserve rerun package.
- **@nprime06** - PR #1787 base architecture.
- **@romeerp** - CaseOps SP8192.
- **@dexhunter** - SmearGate and LQER asymmetric quantization lineage.
- **@cocohearts** - BOS document-boundary bug identification.
- **@abaybektursun** - score-first TTT framework.
- **@clarkkev** - GPTQ and SP8192 lineage.

## Included Files

- `README.md` - this documentation file.
- `train_gpt.py` - SmearGate SOTA with banked train-time LoRA.
- `estimate_lora_param_savings.py` - helper for estimating compact LoRA state size.
