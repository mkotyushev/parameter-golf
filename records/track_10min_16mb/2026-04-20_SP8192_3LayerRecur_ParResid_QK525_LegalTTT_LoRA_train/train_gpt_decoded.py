import collections
import copy
import glob
import io
import lzma
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import uuid

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

from flash_attn_interface import flash_attn_func as flash_attn_3_func


COMPILE_ENABLED_ENV = bool(int(os.environ.get("COMPILE_ENABLED", "1")))
GPTQ_MIN_NUMEL = 65536
LORA_GPTQ_MIN_NUMEL = 32768


class Hyperparameters:
    data_dir = os.environ.get("DATA_DIR", "./data/")
    seed = int(os.environ.get("SEED", 1337))
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))

    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_frac = float(os.environ.get("WARMDOWN_FRAC", 0.72))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 786432))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 500))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 6e2))

    val_batch_tokens = int(os.environ.get("VAL_BATCH_TOKENS", 524288))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 2048))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 4000))
    sliding_window_enabled = bool(int(os.environ.get("SLIDING_WINDOW_ENABLED", "1")))

    vocab_size = int(os.environ.get("VOCAB_SIZE", 8192))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 11))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    embedding_dim = int(os.environ.get("EMBEDDING_DIM", 512))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = float(os.environ.get("MLP_MULT", 4.0))
    skip_gates_enabled = bool(int(os.environ.get("SKIP_GATES_ENABLED", "1")))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 3e1))
    rope_base = float(os.environ.get("ROPE_BASE", 1e4))
    rope_dims = int(os.environ.get("ROPE_DIMS", 16))
    rope_train_seq_len = int(os.environ.get("ROPE_TRAIN_SEQ_LEN", 2048))
    ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 5.0))
    lora_rank = int(os.environ.get("LORA_RANK", 0))
    compile_enabled = bool(int(os.environ.get("COMPILE_ENABLED", "1")))

    num_loops = int(os.environ.get("NUM_LOOPS", 2))
    loop_start = int(os.environ.get("LOOP_START", 3))
    loop_end = int(os.environ.get("LOOP_END", 5))
    enable_looping_at = float(os.environ.get("ENABLE_LOOPING_AT", 0.35))
    parallel_residual_start = int(os.environ.get("PARALLEL_RESIDUAL_START", 7))

    min_lr = float(os.environ.get("MIN_LR", 0.0))
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.03))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.022))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.02))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    muon_row_normalize = bool(int(os.environ.get("MUON_ROW_NORMALIZE", "1")))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    muon_beta2 = float(os.environ.get("MUON_BETA2", 0.95))
    adam_wd = float(os.environ.get("ADAM_WD", 0.02))
    muon_wd = float(os.environ.get("MUON_WD", 0.095))
    embed_wd = float(os.environ.get("EMBED_WD", 0.085))
    ema_decay = float(os.environ.get("EMA_DECAY", 0.9965))

    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "0")))
    ttt_lr = float(os.environ.get("TTT_LR", 0.005))
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 3))
    ttt_momentum = float(os.environ.get("TTT_MOMENTUM", 0.9))
    ttt_chunk_tokens = int(os.environ.get("TTT_CHUNK_TOKENS", 32768))

    etlb_enabled = bool(int(os.environ.get("ETLB_ENABLED", "0")))
    etlb_lr = float(os.environ.get("ETLB_LR", 0.05))
    etlb_steps = int(os.environ.get("ETLB_STEPS", 5))
    etlb_clip = float(os.environ.get("ETLB_CLIP", 3.0))

    compressor = os.environ.get("COMPRESSOR", "brotli")
    gptq_calibration_batches = int(os.environ.get("GPTQ_CALIBRATION_BATCHES", 64))
    gptq_reserve_seconds = float(os.environ.get("GPTQ_RESERVE_SECONDS", 12.0))
    matrix_bits = int(os.environ.get("MATRIX_BITS", 6))
    embed_bits = int(os.environ.get("EMBED_BITS", 8))
    matrix_clip_sigmas = float(os.environ.get("MATRIX_CLIP_SIGMAS", 12.85))
    embed_clip_sigmas = float(os.environ.get("EMBED_CLIP_SIGMAS", 20.0))

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main_process = rank == 0
    grad_accum_steps = 8 // world_size

    datasets_dir = os.path.join(data_dir, "datasets", f"fineweb10B_sp{vocab_size}")
    train_files = os.path.join(datasets_dir, "fineweb_train_*.bin")
    val_files = os.path.join(datasets_dir, "fineweb_val_*.bin")
    tokenizer_path = os.path.join(data_dir, "tokenizers", f"fineweb_{vocab_size}_bpe.model")

    logfile = f"logs/{run_id}.txt"
    model_path = "final_model.pt"
    quantized_model_path = "final_model.int6.ptz"


_logger_hparams = None


def set_logging_hparams(h: Hyperparameters) -> None:
    global _logger_hparams
    _logger_hparams = h


def log(msg, console: bool = True) -> None:
    if _logger_hparams is None:
        print(msg)
        return
    if _logger_hparams.is_main_process:
        if console:
            print(msg)
        if _logger_hparams.logfile is not None:
            with open(_logger_hparams.logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)


def maybe_compile(target, enabled: bool, **kwargs):
    if not enabled or not hasattr(torch, "compile"):
        return target
    return torch.compile(target, **kwargs)


def reset_dynamo() -> None:
    if hasattr(torch, "_dynamo"):
        torch._dynamo.reset()


class ValidationData:
    def __init__(self, h: Hyperparameters, device: torch.device):
        self.sp = spm.SentencePieceProcessor(model_file=h.tokenizer_path)
        if int(self.sp.vocab_size()) != h.vocab_size:
            raise ValueError(
                f"VOCAB_SIZE={h.vocab_size} does not match tokenizer vocab_size={int(self.sp.vocab_size())}"
            )
        self.val_tokens = load_validation_tokens(h.val_files, h.eval_seq_len)
        (
            self.base_bytes_lut,
            self.has_leading_space_lut,
            self.is_boundary_token_lut,
        ) = build_sentencepiece_luts(self.sp, h.vocab_size, device)


def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    assert sp.piece_to_id("\u2581") != sp.unk_id(), (
        "Tokenizer must have '\\u2581' (space) as its own token for correct BPB byte counting"
    )
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


_SHARD_HEADER_BYTES = 256 * np.dtype("<i4").itemsize
_SHARD_NTOKENS_CACHE: dict[str, int] = {}
_MMAP_CACHE: dict[str, np.memmap] = {}


def _read_num_tokens(file: Path) -> int:
    key = str(file)
    cached = _SHARD_NTOKENS_CACHE.get(key)
    if cached is not None:
        return cached
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    _SHARD_NTOKENS_CACHE[key] = num_tokens
    return num_tokens


def _get_shard_memmap(file: Path) -> np.memmap:
    key = str(file)
    mm = _MMAP_CACHE.get(key)
    if mm is not None:
        return mm
    num_tokens = _read_num_tokens(file)
    mm = np.memmap(file, mode="r", dtype="<u2", offset=_SHARD_HEADER_BYTES, shape=(num_tokens,))
    _MMAP_CACHE[key] = mm
    return mm


class ShuffledSequenceLoader:
    def __init__(self, h: Hyperparameters, device: torch.device):
        self.world_size = h.world_size
        self.seq_len = h.train_seq_len
        self.device = device
        all_files = [Path(p) for p in sorted(glob.glob(h.train_files))]
        if not all_files:
            raise FileNotFoundError(f"No files found for pattern: {h.train_files}")
        self.files = all_files[h.rank :: h.world_size]
        self.rng = np.random.Generator(np.random.PCG64(h.rank))
        self.num_tokens = [_read_num_tokens(file) for file in self.files]
        self.start_inds = [[] for _ in self.files]
        for shard_index in range(len(self.files)):
            self._reset_shard(shard_index)

    def _reset_shard(self, shard_index: int) -> None:
        max_phase = min(self.seq_len - 1, max(0, self.num_tokens[shard_index] - self.seq_len - 1))
        phase = int(self.rng.integers(max_phase + 1)) if max_phase > 0 else 0
        num_sequences = (self.num_tokens[shard_index] - 1 - phase) // self.seq_len
        sequence_order = self.rng.permutation(num_sequences)
        self.start_inds[shard_index] = (phase + sequence_order * self.seq_len).tolist()

    def next_batch(self, global_tokens: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        device_tokens = global_tokens // (self.world_size * grad_accum_steps)
        device_batch_size = device_tokens // self.seq_len
        remaining = np.array([len(indices) for indices in self.start_inds], dtype=np.float64)
        x = torch.empty((device_batch_size, self.seq_len), dtype=torch.int64)
        y = torch.empty((device_batch_size, self.seq_len), dtype=torch.int64)
        for batch_index in range(device_batch_size):
            total = remaining.sum()
            if total <= 0:
                for shard_index in range(len(self.files)):
                    self._reset_shard(shard_index)
                remaining = np.array([len(indices) for indices in self.start_inds], dtype=np.float64)
                total = remaining.sum()
            probs = remaining / total
            shard_index = int(self.rng.choice(len(self.files), p=probs))
            start_ind = self.start_inds[shard_index].pop()
            remaining[shard_index] -= 1
            mm = _get_shard_memmap(self.files[shard_index])
            window = torch.as_tensor(np.array(mm[start_ind : start_ind + self.seq_len + 1], dtype=np.int64))
            x[batch_index] = window[:-1]
            y[batch_index] = window[1:]
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


class RMSNorm(nn.Module):
    def __init__(self, eps=None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        weight = self.weight.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)


class LoRALinear(nn.Module):
    def __init__(self, linear: CastedLinear, rank: int, generator: torch.Generator):
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"Expected nn.Linear, got {type(linear).__name__}")
        if rank < 1:
            raise ValueError(f"LORA_RANK must be >= 1, got {rank}")
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.rank = rank
        self.scaling = 1.0
        self.weight = linear.weight
        self.bias = linear.bias
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        self.lora_A = nn.Parameter(self.weight.new_empty((rank, self.in_features)))
        self.lora_B = nn.Parameter(self.weight.new_zeros((self.out_features, rank)))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5), generator=generator)
        nn.init.zeros_(self.lora_B)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"rank={self.rank}"
        )

    def fused_weight(self) -> Tensor:
        return self.weight + (self.lora_B @ self.lora_A) * self.scaling

    def forward(self, x: Tensor) -> Tensor:
        weight = self.weight.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        lora_a = self.lora_A.to(x.dtype)
        lora_b = self.lora_B.to(x.dtype)
        return (
            F.linear(x, weight, bias)
            + F.linear(F.linear(x, lora_a, bias=None), lora_b, bias=None) * self.scaling
        )


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 1e4, train_seq_len: int = 1024, rope_dims: int = 0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.train_seq_len = train_seq_len
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / base ** (
            torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            rope_dims = self.rope_dims
            if seq_len > self.train_seq_len:
                scale = seq_len / self.train_seq_len
                new_base = self.base * scale ** (rope_dims / (rope_dims - 2))
                inv_freq = 1.0 / new_base ** (
                    torch.arange(0, rope_dims, 2, dtype=torch.float32, device=device) / rope_dims
                )
            else:
                inv_freq = self.inv_freq.to(device)
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = freqs.cos()[None, :, None, :]
            self._sin_cached = freqs.sin()[None, :, None, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor, rope_dims: int = 0) -> Tensor:
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rope = torch.cat((x1 * cos + x2 * sin, x1 * -sin + x2 * cos), dim=-1)
        return torch.cat((x_rope, x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * -sin + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        train_seq_len: int,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rope_dims = 0
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=train_seq_len)
        self.use_xsa = False

    def _xsa_efficient(self, y: Tensor, v: Tensor) -> Tensor:
        batch, seqlen, num_heads, head_dim = y.shape
        num_kv_heads = v.size(-2)
        group = num_heads // num_kv_heads
        y_grouped = y.reshape(batch, seqlen, num_kv_heads, group, head_dim)
        v_normalized = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_grouped * v_normalized).sum(dim=-1, keepdim=True) * v_normalized
        return (y_grouped - proj).reshape(batch, seqlen, num_heads, head_dim)

    def forward(self, x: Tensor) -> Tensor:
        batch, seqlen, dim = x.shape
        q = self.c_q(x).reshape(batch, seqlen, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(batch, seqlen, self.num_kv_heads, self.head_dim)
        v = self.c_v(x).reshape(batch, seqlen, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        y = flash_attn_3_func(q, k, v, causal=True)
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        y = y.reshape(batch, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: float):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(F.leaky_relu(self.fc(x), negative_slope=0.5).square())


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: float,
        rope_base: float,
        qk_gain_init: float,
        train_seq_len: int,
        layer_idx: int = 0,
        ln_scale: bool = False,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(
            dim,
            num_heads,
            num_kv_heads,
            rope_base,
            qk_gain_init,
            train_seq_len,
        )
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0
        self.parallel = False

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x_in) * self.ln_scale_factor)
        if self.parallel:
            mlp_out = self.mlp(self.mlp_norm(x_in) * self.ln_scale_factor)
            return (
                x_in
                + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
                + self.mlp_scale.to(dtype=x_in.dtype)[None, None, :] * mlp_out
            )
        x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(
            self.mlp_norm(x_out) * self.ln_scale_factor
        )
        return x_out


class GPT(nn.Module):
    def __init__(self, h: Hyperparameters):
        super().__init__()
        if h.logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {h.logit_softcap}")
        self.tie_embeddings = h.tie_embeddings
        self.tied_embed_init_std = h.tied_embed_init_std
        self.logit_softcap = h.logit_softcap
        self.tok_emb = nn.Embedding(h.vocab_size, h.embedding_dim)
        if h.embedding_dim != h.model_dim:
            self.embed_proj = CastedLinear(h.embedding_dim, h.model_dim, bias=False)
            self.head_proj = CastedLinear(h.model_dim, h.embedding_dim, bias=False)
        else:
            self.embed_proj = None
            self.head_proj = None
        self.num_encoder_layers = h.num_layers // 2
        self.num_decoder_layers = h.num_layers - self.num_encoder_layers
        self.blocks = nn.ModuleList(
            [
                Block(
                    h.model_dim,
                    h.num_heads,
                    h.num_kv_heads,
                    h.mlp_mult,
                    h.rope_base,
                    h.qk_gain_init,
                    h.train_seq_len,
                    layer_idx=i,
                    ln_scale=h.ln_scale,
                )
                for i in range(h.num_layers)
            ]
        )
        if h.rope_dims > 0:
            head_dim = h.model_dim // h.num_heads
            for block in self.blocks:
                block.attn.rope_dims = h.rope_dims
                block.attn.rotary = Rotary(
                    head_dim,
                    base=h.rope_base,
                    train_seq_len=h.train_seq_len,
                    rope_dims=h.rope_dims,
                )
        self.final_norm = RMSNorm()
        self.lm_head = None if h.tie_embeddings else CastedLinear(h.embedding_dim, h.vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        if h.xsa_last_n > 0:
            for layer_idx in range(max(0, h.num_layers - h.xsa_last_n), h.num_layers):
                self.blocks[layer_idx].attn.use_xsa = True
        if h.parallel_residual_start >= 0:
            for layer_idx in range(h.parallel_residual_start, h.num_layers):
                self.blocks[layer_idx].parallel = True
        self.looping_active = False
        if h.num_loops > 0:
            loop_seg = list(range(h.loop_start, h.loop_end + 1))
            all_indices = list(range(h.loop_start))
            for _ in range(h.num_loops + 1):
                all_indices.extend(loop_seg)
            all_indices.extend(range(h.loop_end + 1, h.num_layers))
            num_encoder = len(all_indices) // 2
            self.encoder_indices = all_indices[:num_encoder]
            self.decoder_indices = all_indices[num_encoder:]
        else:
            self.encoder_indices = list(range(self.num_encoder_layers))
            self.decoder_indices = list(range(self.num_encoder_layers, h.num_layers))
        self.num_skip_weights = min(len(self.encoder_indices), len(self.decoder_indices))
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, h.model_dim, dtype=torch.float32))
        self.skip_gates = (
            nn.Parameter(torch.zeros(self.num_skip_weights, h.model_dim, dtype=torch.float32))
            if h.skip_gates_enabled
            else None
        )
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for _, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif (
                    module.weight.ndim == 2
                    and module.weight.shape[0] >= 64
                    and module.weight.shape[1] >= 64
                ):
                    nn.init.orthogonal_(module.weight, gain=1.0)

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        if self.embed_proj is not None:
            x = self.embed_proj(x)
        x0 = x
        skips = []
        encoder_iter = self.encoder_indices if self.looping_active else range(self.num_encoder_layers)
        decoder_iter = (
            self.decoder_indices
            if self.looping_active
            else range(self.num_encoder_layers, self.num_encoder_layers + self.num_decoder_layers)
        )
        for layer_idx in encoder_iter:
            x = self.blocks[layer_idx](x, x0)
            skips.append(x)
        for skip_idx, layer_idx in enumerate(decoder_iter):
            if skip_idx < self.num_skip_weights and skips:
                scaled_skip = self.skip_weights[skip_idx].to(dtype=x.dtype)[None, None, :] * skips.pop()
                if self.skip_gates is not None:
                    gates = torch.sigmoid(self.skip_gates[skip_idx].to(dtype=x.dtype))[None, None, :]
                    x = torch.lerp(scaled_skip, x, gates)
                else:
                    x = x + scaled_skip
            x = self.blocks[layer_idx](x, x0)
        x = self.final_norm(x)
        if self.head_proj is not None:
            x = self.head_proj(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        logits = self.forward_logits(input_ids)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            target_ids.reshape(-1),
            reduction="mean",
        )


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def snapshot_rng_state() -> dict[str, object]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def make_lora_generator(seed: int, device: torch.device) -> torch.Generator:
    generator_device = "cuda" if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    return generator


def replace_casted_linear_with_lora(module: nn.Module, rank: int, generator: torch.Generator) -> nn.Module:
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            continue
        if isinstance(child, CastedLinear):
            setattr(module, name, LoRALinear(child, rank=rank, generator=generator))
            continue
        replace_casted_linear_with_lora(child, rank=rank, generator=generator)
    return module


def restore_fp32_params(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (CastedLinear, LoRALinear)):
            module.float()
    for name, param in model.named_parameters():
        if (param.ndim < 2 or is_control_tensor_name(name)) and param.dtype != torch.float32:
            param.data = param.data.float()


def clone_hparams(h: Hyperparameters, **updates) -> Hyperparameters:
    cloned = copy.copy(h)
    for key, value in updates.items():
        setattr(cloned, key, value)
    return cloned


def build_model(
    h: Hyperparameters,
    device: torch.device,
    init_seed: int | None = None,
    lora_rank: int | None = None,
) -> GPT:
    init_seed = h.seed if init_seed is None else init_seed
    lora_rank = h.lora_rank if lora_rank is None else lora_rank
    rng_state = snapshot_rng_state()
    try:
        seed_all(init_seed)
        model = GPT(h).to(device).bfloat16()
        restore_fp32_params(model)
        if lora_rank > 0:
            lora_generator = make_lora_generator(init_seed + 1, device)
            replace_casted_linear_with_lora(model, rank=lora_rank, generator=lora_generator)
        return model
    finally:
        restore_rng_state(rng_state)


def has_lora_layers(model: nn.Module) -> bool:
    return any(isinstance(module, LoRALinear) for module in model.modules())


def get_lora_rank(model: nn.Module) -> int:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            return module.rank
    return 0


def get_lora_parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for name, param in model.named_parameters() if "lora_" in name)


def gptq_min_numel_for_name(name: str) -> int:
    return LORA_GPTQ_MIN_NUMEL if "lora_" in name else GPTQ_MIN_NUMEL


def should_gptq_named_tensor(name: str, tensor: Tensor) -> bool:
    return tensor.is_floating_point() and tensor.numel() >= gptq_min_numel_for_name(name)


def get_reconstructible_tensor_names(model: nn.Module) -> set[str]:
    names = set()
    for module_name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        prefix = f"{module_name}." if module_name else ""
        names.add(f"{prefix}weight")
        if module.bias is not None:
            names.add(f"{prefix}bias")
    return names


def build_fused_state_dict(model: nn.Module) -> collections.OrderedDict[str, Tensor]:
    lora_layers = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
    }
    fused_state = collections.OrderedDict()
    for key, value in model.state_dict().items():
        module_name, _, param_name = key.rpartition(".")
        if param_name in {"lora_A", "lora_B"}:
            continue
        if param_name == "weight" and module_name in lora_layers:
            fused_state[key] = lora_layers[module_name].fused_weight().detach().clone()
            continue
        fused_state[key] = value
    return fused_state


def build_compact_state_dict(model: nn.Module) -> collections.OrderedDict[str, Tensor]:
    reconstructible = get_reconstructible_tensor_names(model)
    compact_state = collections.OrderedDict()
    for key, value in model.state_dict().items():
        if key in reconstructible:
            continue
        compact_state[key] = value
    return compact_state


def build_serializable_state_dict(model: nn.Module) -> collections.OrderedDict[str, Tensor]:
    if has_lora_layers(model):
        return build_compact_state_dict(model)
    return collections.OrderedDict((key, value) for key, value in model.state_dict().items())


def state_dict_to_cpu(state_dict: collections.OrderedDict[str, Tensor] | dict[str, Tensor]) -> dict[str, Tensor]:
    return {key: value.detach().cpu() for key, value in state_dict.items()}


def classify_param(name: str) -> str:
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name or (".proj." in name and ".mlp." not in name):
        return "attn"
    return "other"


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = 3.4445, -4.775, 2.0315
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


if COMPILE_ENABLED_ENV and hasattr(torch, "compile"):
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        momentum: float,
        backend_steps: int,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        row_normalize: bool = False,
    ):
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                backend_steps=backend_steps,
                nesterov=nesterov,
                weight_decay=weight_decay,
                row_normalize=row_normalize,
            ),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            total_params = sum(int(param.numel()) for param in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for index, param in enumerate(params):
                if index % world_size == rank and param.grad is not None:
                    grad = param.grad
                    state = self.state[param]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad)
                    if nesterov:
                        grad = grad.add(buf, alpha=momentum)
                    if group.get("row_normalize", False):
                        row_norms = grad.float().norm(dim=-1, keepdim=True).clamp_min(1e-7)
                        grad = grad / row_norms.to(grad.dtype)
                    grad = zeropower_via_newtonschulz5(grad, steps=backend_steps)
                    grad *= max(1, grad.size(0) / grad.size(1)) ** 0.5
                    updates_flat[curr : curr + param.numel()] = grad.reshape(-1)
                curr += param.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            weight_decay = group.get("weight_decay", 0.0)
            curr = 0
            for param in params:
                if weight_decay > 0.0:
                    param.data.mul_(1.0 - lr * weight_decay)
                grad = updates_flat[curr : curr + param.numel()].view_as(param).to(dtype=param.dtype)
                param.add_(grad, alpha=-lr)
                curr += param.numel()
        return loss


CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,skip_gates",
    ).split(",")
    if pattern
)


def is_control_tensor_name(name: str) -> bool:
    return any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)


class Optimizers:
    def __init__(self, h: Hyperparameters, base_model: GPT):
        fused = True
        token_lr = h.tied_embed_lr if h.tie_embeddings else h.embed_lr

        tok_params = []
        head_params = []
        matrix_params = []
        scalar_params = []

        for name, param in base_model.named_parameters():
            if not param.requires_grad:
                continue
            if name == "tok_emb.weight":
                tok_params.append(param)
                continue
            if name.startswith("lm_head."):
                head_params.append(param)
                continue
            if param.ndim == 2 and not is_control_tensor_name(name):
                matrix_params.append(param)
                continue
            scalar_params.append(param)

        self.optimizer_tok = torch.optim.AdamW(
            [{"params": tok_params, "lr": token_lr, "base_lr": token_lr}],
            betas=(h.beta1, h.beta2),
            eps=h.adam_eps,
            weight_decay=h.embed_wd,
            fused=fused,
        ) if tok_params else None

        self.optimizer_head = torch.optim.Adam(
            [{"params": head_params, "lr": h.head_lr, "base_lr": h.head_lr}],
            betas=(h.beta1, h.beta2),
            eps=h.adam_eps,
            fused=fused,
        ) if head_params else None

        self.optimizer_muon = Muon(
            matrix_params,
            lr=h.matrix_lr,
            momentum=h.muon_momentum,
            backend_steps=h.muon_backend_steps,
            weight_decay=h.muon_wd,
            row_normalize=h.muon_row_normalize,
        ) if matrix_params else None
        if self.optimizer_muon is not None:
            for group in self.optimizer_muon.param_groups:
                group["base_lr"] = h.matrix_lr

        self.optimizer_scalar = torch.optim.AdamW(
            [{"params": scalar_params, "lr": h.scalar_lr, "base_lr": h.scalar_lr}],
            betas=(h.beta1, h.beta2),
            eps=h.adam_eps,
            weight_decay=h.adam_wd,
            fused=fused,
        ) if scalar_params else None

        self.optimizers = [
            optimizer
            for optimizer in (
                self.optimizer_tok,
                self.optimizer_head,
                self.optimizer_muon,
                self.optimizer_scalar,
            )
            if optimizer is not None
        ]

    def __iter__(self):
        return iter(self.optimizers)

    def zero_grad_all(self) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=True)

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()
        self.zero_grad_all()


def collect_hessians(
    model: GPT,
    train_loader: ShuffledSequenceLoader,
    h: Hyperparameters,
    device: torch.device,
    n_calibration_batches: int = 64,
) -> dict[str, Tensor]:
    hessians = {}
    hooks = []

    def flatten_input(x: Tensor) -> Tensor:
        x = x.detach().float()
        if x.ndim == 3:
            x = x.reshape(-1, x.shape[-1])
        return x

    def add_hessian(name: str, x: Tensor) -> None:
        if name not in hessians:
            hessians[name] = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32, device=device)
        hessians[name].addmm_(x.T, x)

    def make_hook(name: str):
        def hook_fn(module, inp, out):
            add_hessian(name, flatten_input(inp[0]))

        return hook_fn

    def make_lora_hook(name: str, quantize_a: bool, quantize_b: bool):
        def hook_fn(module, inp, out):
            x = flatten_input(inp[0])
            if quantize_a:
                add_hessian(name + ".lora_A", x)
            if quantize_b:
                projected = F.linear(x, module.lora_A.detach().float(), bias=None)
                add_hessian(name + ".lora_B", projected)

        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, CastedLinear) and should_gptq_named_tensor(name + ".weight", module.weight):
            category = classify_param(name + ".weight")
            if category in ("mlp", "attn"):
                hooks.append(module.register_forward_hook(make_hook(name + ".weight")))
        elif isinstance(module, LoRALinear):
            quantize_a = should_gptq_named_tensor(name + ".lora_A", module.lora_A)
            quantize_b = should_gptq_named_tensor(name + ".lora_B", module.lora_B)
            if quantize_a or quantize_b:
                hooks.append(module.register_forward_hook(make_lora_hook(name, quantize_a, quantize_b)))

    if model.tie_embeddings:
        hook_module = model.head_proj if model.head_proj is not None else model.final_norm

        def make_output_hook(name: str):
            def hook_fn(module, inp, out):
                x = out.detach().float()
                if x.ndim == 3:
                    x = x.reshape(-1, x.shape[-1])
                if name not in hessians:
                    hessians[name] = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32, device=device)
                hessians[name].addmm_(x.T, x)

            return hook_fn

        hooks.append(hook_module.register_forward_hook(make_output_hook("tok_emb.weight")))

    model.eval()
    with torch.no_grad():
        for _ in range(n_calibration_batches):
            x, _ = train_loader.next_batch(h.train_batch_tokens, h.grad_accum_steps)
            model.forward_logits(x)
    for hook in hooks:
        hook.remove()
    for name in hessians:
        hessians[name] = hessians[name].cpu() / n_calibration_batches
    return hessians


def gptq_quantize_weight(
    weight: Tensor,
    hessian: Tensor,
    clip_sigmas: float = 3.0,
    clip_range: int = 63,
    block_size: int = 128,
) -> tuple[Tensor, Tensor]:
    W_orig = weight.float().clone()
    rows, cols = W_orig.shape
    H = hessian.float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    damp = 0.01 * H.diag().mean()
    H.diagonal().add_(damp)
    perm = torch.argsort(H.diag(), descending=True)
    invperm = torch.argsort(perm)
    W_perm = W_orig[:, perm].clone()
    W_perm[:, dead[perm]] = 0
    H = H[perm][:, perm]
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    row_std = W_orig.std(dim=1)
    scale = (clip_sigmas * row_std / clip_range).clamp_min(1e-10).to(torch.float16)
    scale_float = scale.float()
    Q = torch.zeros(rows, cols, dtype=torch.int8)
    W_work = W_perm.clone()
    for start in range(0, cols, block_size):
        end = min(start + block_size, cols)
        W_block = W_work[:, start:end].clone()
        Hinv_block = Hinv[start:end, start:end]
        err = torch.zeros(rows, end - start)
        for column in range(end - start):
            w_col = W_block[:, column]
            denom = Hinv_block[column, column]
            q_col = torch.clamp(torch.round(w_col / scale_float), -clip_range, clip_range)
            Q[:, start + column] = q_col.to(torch.int8)
            err[:, column] = (w_col - q_col.float() * scale_float) / denom
            W_block[:, column:] -= err[:, column].unsqueeze(1) * Hinv_block[column, column:].unsqueeze(0)
        if end < cols:
            W_work[:, end:] -= err @ Hinv[start:end, end:]
    return Q[:, invperm], scale


def gptq_mixed_quantize(
    state_dict: dict[str, Tensor],
    hessians: dict[str, Tensor],
    h: Hyperparameters,
) -> tuple[dict[str, Tensor], dict[str, str]]:
    result = {}
    meta = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        if not t.is_floating_point():
            result[name] = t
            meta[name] = "passthrough"
            continue
        if t.numel() < gptq_min_numel_for_name(name):
            result[name] = t.to(torch.float16)
            meta[name] = "passthrough (float16)"
            continue
        if name not in hessians:
            result[name] = t.to(torch.float16)
            meta[name] = "passthrough (float16, no_hessian)"
            continue
        clip_sigmas = h.embed_clip_sigmas if "tok_emb" in name else h.matrix_clip_sigmas
        bits = h.embed_bits if "tok_emb" in name else h.matrix_bits
        q, scale = gptq_quantize_weight(
            t,
            hessians[name],
            clip_sigmas=clip_sigmas,
            clip_range=2 ** (bits - 1) - 1,
        )
        result[name + ".q"] = q
        result[name + ".scale"] = scale
        meta[name] = f"gptq (int{bits})"
    categories = collections.defaultdict(set)
    for name, category in meta.items():
        short = re.sub(r"\.\d+$", "", re.sub(r"blocks\.\d+", "blocks", name))
        categories[category].add(short)
    log("Quantized weights:")
    for category in sorted(categories):
        joined = ", ".join(sorted(categories[category]))
        log(f"  {category}: {joined}")
    return result, meta


def dequantize_mixed(
    result: dict[str, Tensor],
    meta: dict[str, str],
    template_sd: dict[str, Tensor],
) -> dict[str, Tensor]:
    out = {}
    for name, original in template_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        orig_dtype = original.dtype
        if "passthrough" in info:
            tensor = result[name]
            if tensor.dtype == torch.float16 and orig_dtype in (torch.float32, torch.bfloat16):
                tensor = tensor.to(orig_dtype)
            out[name] = tensor
            continue
        q = result[name + ".q"]
        scale = result[name + ".scale"]
        if scale.ndim > 0:
            out[name] = (q.float() * scale.float().view(q.shape[0], *[1] * (q.ndim - 1))).to(orig_dtype)
        else:
            out[name] = (q.float() * float(scale.item())).to(orig_dtype)
    return out


_BSHF_MAGIC = b"BSHF"


def _byte_shuffle(data: bytes, stride: int = 2) -> bytes:
    if stride <= 1 or len(data) < stride:
        return data
    src = np.frombuffer(data, dtype=np.uint8)
    n = len(src)
    out = np.empty(n, dtype=np.uint8)
    dest_off = 0
    for pos in range(stride):
        chunk = src[pos::stride]
        out[dest_off : dest_off + len(chunk)] = chunk
        dest_off += len(chunk)
    return _BSHF_MAGIC + bytes([stride]) + out.tobytes()


def _byte_unshuffle(data: bytes) -> bytes:
    if len(data) < 5 or data[:4] != _BSHF_MAGIC:
        return data
    stride = data[4]
    if stride < 2:
        return data[5:]
    payload = np.frombuffer(data, dtype=np.uint8, offset=5)
    n = len(payload)
    out = np.empty(n, dtype=np.uint8)
    src_off = 0
    for pos in range(stride):
        chunk_len = n // stride + (1 if pos < n % stride else 0)
        out[pos::stride][:chunk_len] = payload[src_off : src_off + chunk_len]
        src_off += chunk_len
    return out.tobytes()


def _compress(data: bytes, compressor: str) -> bytes:
    data = _byte_shuffle(data)
    if compressor == "lzma":
        return lzma.compress(data, preset=6)
    if compressor == "brotli":
        import brotli

        return brotli.compress(data, quality=11)
    raise ValueError(f"Unknown compressor: {compressor!r}")


def _decompress(data: bytes, compressor: str) -> bytes:
    if compressor == "lzma":
        raw = lzma.decompress(data)
    elif compressor == "brotli":
        import brotli

        raw = brotli.decompress(data)
    else:
        raise ValueError(f"Unknown compressor: {compressor!r}")
    return _byte_unshuffle(raw)


def build_artifact_meta(h: Hyperparameters, model: nn.Module) -> dict[str, object]:
    artifact_mode = "lora" if has_lora_layers(model) else "full"
    lora_rank = get_lora_rank(model) if artifact_mode == "lora" else 0
    return {
        "artifact_mode": artifact_mode,
        "init_seed": h.seed,
        "lora_rank": lora_rank,
    }


def serialize(h: Hyperparameters, base_model: GPT, code: str) -> tuple[int, int]:
    code_bytes = len(code.encode("utf-8"))
    serializable_state = build_serializable_state_dict(base_model)
    if h.is_main_process:
        torch.save(serializable_state, h.model_path)
        model_bytes = os.path.getsize(h.model_path)
        log(f"Serialized model: {model_bytes} bytes")
        log(f"Code size: {code_bytes} bytes")
        if has_lora_layers(base_model):
            omitted = len(get_reconstructible_tensor_names(base_model))
            log(f"LoRA artifact mode: compact ({omitted} reconstructible tensors omitted)")
    sd_cpu = state_dict_to_cpu(serializable_state)
    device = torch.device("cuda", h.local_rank)
    log("GPTQ:collecting Hessians from calibration data...")
    t0 = time.perf_counter()
    calib_loader = ShuffledSequenceLoader(h, device)
    hessians = collect_hessians(
        base_model,
        calib_loader,
        h,
        device,
        n_calibration_batches=h.gptq_calibration_batches,
    )
    log(f"GPTQ:collected {len(hessians)} Hessians in {time.perf_counter() - t0:.1f}s")
    quant_result, quant_meta = gptq_mixed_quantize(sd_cpu, hessians, h)
    artifact_meta = build_artifact_meta(h, base_model)
    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta, "artifact_meta": artifact_meta}, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = _compress(quant_raw, h.compressor)
    quant_file_bytes = len(quant_blob)
    bytes_total = quant_file_bytes + code_bytes
    if h.is_main_process:
        with open(h.quantized_model_path, "wb") as f:
            f.write(quant_blob)
        log(f"Serialized model quantized+{h.compressor}: {quant_file_bytes} bytes")
        log(f"Total submission size quantized+{h.compressor}: {bytes_total} bytes")
    return bytes_total, quant_file_bytes


def _load_quantized_artifact(h: Hyperparameters) -> dict[str, object]:
    with open(h.quantized_model_path, "rb") as f:
        quant_blob_disk = f.read()
    return torch.load(io.BytesIO(_decompress(quant_blob_disk, h.compressor)), map_location="cpu")


def _resolve_artifact_meta(h: Hyperparameters, quant_state: dict[str, object]) -> dict[str, object]:
    artifact_meta = quant_state.get("artifact_meta", None)
    if artifact_meta is None:
        artifact_meta = {
            "artifact_mode": "full",
            "init_seed": h.seed,
            "lora_rank": 0,
        }
    artifact_mode = artifact_meta.get("artifact_mode", "full")
    init_seed = int(artifact_meta.get("init_seed", h.seed))
    lora_rank = int(artifact_meta.get("lora_rank", 0))
    if artifact_mode not in {"full", "lora"}:
        raise ValueError(f"Unknown artifact_mode={artifact_mode!r}")
    if artifact_mode == "full" and lora_rank != 0:
        raise ValueError(f"Full artifact must have lora_rank=0, got {lora_rank}")
    if artifact_mode == "lora" and lora_rank < 1:
        raise ValueError(f"LoRA artifact must have lora_rank>=1, got {lora_rank}")
    if h.seed != init_seed:
        log(f"deserialize: using artifact init_seed={init_seed} instead of current SEED={h.seed}")
    if h.lora_rank != lora_rank:
        log(f"deserialize: using artifact lora_rank={lora_rank} instead of current LORA_RANK={h.lora_rank}")
    return {
        "artifact_mode": artifact_mode,
        "init_seed": init_seed,
        "lora_rank": lora_rank,
    }


def deserialize(h: Hyperparameters, device: torch.device) -> GPT:
    quant_state = _load_quantized_artifact(h)
    artifact_meta = _resolve_artifact_meta(h, quant_state)
    artifact_h = clone_hparams(
        h,
        seed=artifact_meta["init_seed"],
        lora_rank=artifact_meta["lora_rank"],
    )

    if artifact_meta["artifact_mode"] == "full":
        eval_model = build_model(artifact_h, device, init_seed=artifact_meta["init_seed"], lora_rank=0)
        template_sd = state_dict_to_cpu(collections.OrderedDict(eval_model.state_dict()))
        deq_state = dequantize_mixed(quant_state["w"], quant_state["m"], template_sd)
        eval_model.load_state_dict(deq_state, strict=True)
        return eval_model

    lora_model = build_model(
        artifact_h,
        device,
        init_seed=artifact_meta["init_seed"],
        lora_rank=artifact_meta["lora_rank"],
    )
    compact_template = state_dict_to_cpu(build_serializable_state_dict(lora_model))
    deq_state = dequantize_mixed(quant_state["w"], quant_state["m"], compact_template)
    missing_keys, unexpected_keys = lora_model.load_state_dict(deq_state, strict=False)
    expected_missing = get_reconstructible_tensor_names(lora_model)
    if unexpected_keys or set(missing_keys) != expected_missing:
        raise RuntimeError(
            "LoRA artifact load mismatch: "
            f"missing={sorted(missing_keys)} unexpected={sorted(unexpected_keys)} "
            f"expected_missing={sorted(expected_missing)}"
        )
    fused_state = build_fused_state_dict(lora_model)
    plain_h = clone_hparams(artifact_h, lora_rank=0)
    eval_model = build_model(plain_h, device, init_seed=artifact_meta["init_seed"], lora_rank=0)
    eval_model.load_state_dict(fused_state, strict=True)
    return eval_model


def _loss_bpb(loss_sum: Tensor, token_count: Tensor, byte_count: Tensor) -> tuple[float, float]:
    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())
    return val_loss, val_bpb


def eval_val(h: Hyperparameters, device: torch.device, val_data: ValidationData, model) -> tuple[float, float]:
    seq_len = h.eval_seq_len
    local_batch_tokens = h.val_batch_tokens // (h.world_size * h.grad_accum_steps)
    if local_batch_tokens < seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={h.val_batch_tokens}, WORLD_SIZE={h.world_size}, "
            f"GRAD_ACCUM_STEPS={h.grad_accum_steps}, seq_len={seq_len}"
        )
    local_batch_seqs = local_batch_tokens // seq_len
    total_seqs = (val_data.val_tokens.numel() - 1) // seq_len
    seq_start = total_seqs * h.rank // h.world_size
    seq_end = total_seqs * (h.rank + 1) // h.world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * seq_len
            raw_end = batch_seq_end * seq_len + 1
            local = val_data.val_tokens[raw_start:raw_end].to(
                device=device,
                dtype=torch.int64,
                non_blocking=True,
            )
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = val_data.base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (
                val_data.has_leading_space_lut[tgt_ids] & ~val_data.is_boundary_token_lut[prev_ids]
            ).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)
    model.train()
    return _loss_bpb(val_loss_sum, val_token_count, val_byte_count)


def eval_val_sliding(
    h: Hyperparameters,
    device: torch.device,
    val_data: ValidationData,
    base_model: GPT,
    batch_seqs: int = 32,
) -> tuple[float, float]:
    base_model.eval()
    logits_fn = maybe_compile(base_model.forward_logits, h.compile_enabled, dynamic=False, fullgraph=True)
    seq_len = h.eval_seq_len
    context_size = seq_len - h.eval_stride
    total_tokens = val_data.val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, h.eval_stride) if ws + context_size < total_tokens]
    total_windows = len(window_starts)
    my_start = total_windows * h.rank // h.world_size
    my_end = total_windows * (h.rank + 1) // h.world_size
    my_windows = window_starts[my_start:my_end]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    with torch.inference_mode():
        for batch_index in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[batch_index : batch_index + batch_seqs]
            batch_size = len(batch_ws)
            x_batch = torch.zeros(batch_size, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(batch_size, seq_len, dtype=torch.int64, device=device)
            window_lengths = []
            for i, ws in enumerate(batch_ws):
                window_end = min(ws + seq_len, total_tokens)
                window_len = window_end - ws
                window_lengths.append(window_len)
                chunk = val_data.val_tokens[ws : window_end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :window_len] = chunk[:-1]
                y_batch[i, :window_len] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = logits_fn(x_batch)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(batch_size, seq_len)
            for i, ws in enumerate(batch_ws):
                window_len = window_lengths[i]
                start = 0 if ws == 0 else context_size
                scored_nll = nll[i, start:window_len].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(window_len - start)
                tgt = y_batch[i, start:window_len]
                prev = x_batch[i, start:window_len]
                token_bytes = val_data.base_bytes_lut[tgt].to(torch.float64)
                token_bytes += (
                    val_data.has_leading_space_lut[tgt] & ~val_data.is_boundary_token_lut[prev]
                ).to(torch.float64)
                byte_count += token_bytes.sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    base_model.train()
    return _loss_bpb(loss_sum, token_count, byte_count)


def eval_val_ttt(
    h: Hyperparameters,
    device: torch.device,
    val_data: ValidationData,
    base_model: GPT,
    batch_seqs: int = 32,
) -> tuple[float, float]:
    rank = h.rank
    world_size = h.world_size
    seq_len = h.eval_seq_len
    stride = h.eval_stride
    total_tokens = val_data.val_tokens.numel() - 1
    ttt_chunk = h.ttt_chunk_tokens
    context_size = seq_len - stride
    window_starts = [ws for ws in range(0, total_tokens, stride) if ws + context_size < total_tokens]
    num_chunks = (total_tokens + ttt_chunk - 1) // ttt_chunk
    chunk_windows = [[] for _ in range(num_chunks)]
    for ws in window_starts:
        window_len = min(ws + seq_len, total_tokens) - ws
        start = 0 if ws == 0 else context_size
        scored_start = ws + start
        chunk_index = min(scored_start // ttt_chunk, num_chunks - 1)
        chunk_windows[chunk_index].append(ws)
    log(f"ttt:start chunks={num_chunks} ttt_lr={h.ttt_lr} ttt_epochs={h.ttt_epochs}")
    compiled_logits = maybe_compile(base_model.forward_logits, h.compile_enabled, dynamic=False, fullgraph=True)
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    ttt_params = [param for param in base_model.parameters()]
    for param in ttt_params:
        param.requires_grad_(True)
    optimizer = torch.optim.SGD(ttt_params, lr=h.ttt_lr, momentum=h.ttt_momentum)
    for chunk_index in range(num_chunks):
        windows = chunk_windows[chunk_index]
        if not windows:
            continue
        chunk_start = chunk_index * ttt_chunk
        chunk_end = min((chunk_index + 1) * ttt_chunk, total_tokens)
        my_start = len(windows) * rank // world_size
        my_end = len(windows) * (rank + 1) // world_size
        my_windows = windows[my_start:my_end]
        base_model.eval()
        with torch.no_grad():
            for batch_index in range(0, len(my_windows), batch_seqs):
                batch_ws = my_windows[batch_index : batch_index + batch_seqs]
                batch_size = len(batch_ws)
                x_batch = torch.zeros(batch_size, seq_len, dtype=torch.int64, device=device)
                y_batch = torch.zeros(batch_size, seq_len, dtype=torch.int64, device=device)
                window_lengths = []
                for i, ws in enumerate(batch_ws):
                    window_end = min(ws + seq_len, total_tokens)
                    window_len = window_end - ws
                    window_lengths.append(window_len)
                    chunk_tokens = val_data.val_tokens[ws : window_end + 1].to(dtype=torch.int64, device=device)
                    x_batch[i, :window_len] = chunk_tokens[:-1]
                    y_batch[i, :window_len] = chunk_tokens[1:]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = compiled_logits(x_batch)
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    y_batch.reshape(-1),
                    reduction="none",
                ).reshape(batch_size, seq_len)
                for i, ws in enumerate(batch_ws):
                    window_len = window_lengths[i]
                    start = 0 if ws == 0 else context_size
                    scored_nll = nll[i, start:window_len].to(torch.float64)
                    loss_sum += scored_nll.sum()
                    token_count += float(window_len - start)
                    tgt = y_batch[i, start:window_len]
                    prev = x_batch[i, start:window_len]
                    token_bytes = val_data.base_bytes_lut[tgt].to(torch.float64)
                    token_bytes += (
                        val_data.has_leading_space_lut[tgt] & ~val_data.is_boundary_token_lut[prev]
                    ).to(torch.float64)
                    byte_count += token_bytes.sum()
        is_last_chunk = chunk_index == num_chunks - 1
        if not is_last_chunk and h.ttt_epochs > 0:
            base_model.train()
            chunk_seqs = (chunk_end - chunk_start) // seq_len
            if chunk_seqs > 0:
                cos_lr = h.ttt_lr * 0.5 * (1.0 + math.cos(math.pi * chunk_index / max(num_chunks - 1, 1)))
                for param_group in optimizer.param_groups:
                    param_group["lr"] = cos_lr
                my_seq_start = chunk_seqs * rank // world_size
                my_seq_end = chunk_seqs * (rank + 1) // world_size
                my_chunk_seqs = my_seq_end - my_seq_start
                for _ in range(h.ttt_epochs):
                    for batch_start in range(0, my_chunk_seqs, batch_seqs):
                        batch_end = min(batch_start + batch_seqs, my_chunk_seqs)
                        actual_bs = my_seq_start + batch_start
                        start_tok = chunk_start + actual_bs * seq_len
                        end_tok = chunk_start + (my_seq_start + batch_end) * seq_len + 1
                        if end_tok > val_data.val_tokens.numel():
                            continue
                        local = val_data.val_tokens[start_tok:end_tok].to(device=device, dtype=torch.int64)
                        x = local[:-1].reshape(-1, seq_len)
                        y = local[1:].reshape(-1, seq_len)
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            loss = base_model(x, y)
                        loss.backward()
                        if world_size > 1:
                            for param in ttt_params:
                                if param.grad is not None:
                                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
                        torch.nn.utils.clip_grad_norm_(ttt_params, 1.0)
                        optimizer.step()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    for param in base_model.parameters():
        param.requires_grad_(True)
    base_model.eval()
    return _loss_bpb(loss_sum, token_count, byte_count)


def timed_eval(label: str, fn, *args, **kwargs) -> tuple[float, float]:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    val_loss, val_bpb = fn(*args, **kwargs)
    torch.cuda.synchronize()
    elapsed_ms = 1e3 * (time.perf_counter() - t0)
    log(f"{label} val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f} eval_time:{elapsed_ms:.0f}ms")
    return val_loss, val_bpb


def train_model(h: Hyperparameters, device: torch.device, val_data: ValidationData) -> tuple[GPT, nn.Module]:
    base_model = build_model(h, device)
    compiled_model = maybe_compile(base_model, h.compile_enabled, dynamic=False, fullgraph=True)
    if h.distributed:
        model = DDP(compiled_model, device_ids=[h.local_rank], broadcast_buffers=False)
    else:
        model = compiled_model

    total_params = sum(param.numel() for param in base_model.parameters())
    trainable_params = sum(param.numel() for param in base_model.parameters() if param.requires_grad)
    lora_params = get_lora_parameter_count(base_model)
    log(
        f"model_params:{total_params} trainable_params:{trainable_params} "
        f"lora_rank:{h.lora_rank} lora_params:{lora_params}"
    )
    optimizers = Optimizers(h, base_model)
    train_loader = ShuffledSequenceLoader(h, device)
    max_wallclock_ms = 1e3 * h.max_wallclock_seconds if h.max_wallclock_seconds > 0 else None
    if max_wallclock_ms is not None:
        max_wallclock_ms -= h.gptq_reserve_seconds * 1e3
        log(f"gptq:reserving {h.gptq_reserve_seconds:.0f}s, effective={max_wallclock_ms:.0f}ms")

    def training_frac(step: int, elapsed_ms: float) -> float:
        if max_wallclock_ms is None:
            return step / max(h.iterations, 1)
        return elapsed_ms / max(max_wallclock_ms, 1e-9)

    def lr_mul(frac: float) -> float:
        if h.warmdown_frac <= 0:
            return 1.0
        if frac >= 1.0 - h.warmdown_frac:
            return max((1.0 - frac) / h.warmdown_frac, h.min_lr)
        return 1.0

    def step_fn(step: int, lr_scale: float) -> Tensor:
        optimizers.zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(h.grad_accum_steps):
            if h.distributed:
                model.require_backward_grad_sync = micro_step == h.grad_accum_steps - 1
            x, y = train_loader.next_batch(h.train_batch_tokens, h.grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss / h.grad_accum_steps).backward()
        train_loss /= h.grad_accum_steps
        frac = min(step / h.muon_momentum_warmup_steps, 1.0) if h.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * h.muon_momentum_warmup_start + frac * h.muon_momentum
        if optimizers.optimizer_muon is not None:
            for group in optimizers.optimizer_muon.param_groups:
                group["momentum"] = muon_momentum
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * lr_scale
        if h.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), h.grad_clip_norm)
        optimizers.step()
        return train_loss

    if h.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(optimizer.state_dict()) for optimizer in optimizers]
        model.train()
        for warmup_step in range(h.warmup_steps):
            step_fn(warmup_step, 1.0)
            if warmup_step <= 5 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == h.warmup_steps:
                log(f"warmup_step: {warmup_step + 1}/{h.warmup_steps}")
        if h.num_loops > 0:
            base_model.looping_active = True
            log(
                f"loop_warmup:enabled encoder:{base_model.encoder_indices} "
                f"decoder:{base_model.decoder_indices}"
            )
            for warmup_step in range(h.warmup_steps):
                step_fn(warmup_step, 1.0)
                if warmup_step <= 5 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == h.warmup_steps:
                    log(f"loop_warmup_step: {warmup_step + 1}/{h.warmup_steps}")
            base_model.looping_active = False
        base_model.load_state_dict(initial_model_state, strict=True)
        for optimizer, state in zip(optimizers, initial_optimizer_states, strict=True):
            optimizer.load_state_dict(state)
        optimizers.zero_grad_all()
        if h.distributed:
            model.require_backward_grad_sync = True
        train_loader = ShuffledSequenceLoader(h, device)

    ema_state = {name: tensor.detach().float().clone() for name, tensor in base_model.state_dict().items()}
    ema_decay = h.ema_decay
    training_time_ms = 0.0
    stop_after_step = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0
    while True:
        last_step = step == h.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (h.val_loss_every > 0 and step % h.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1e3 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(h, device, val_data, model)
            log(f"{step}/{h.iterations} val_loss: {val_loss:.4f} val_bpb: {val_bpb:.4f}")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < h.iterations:
                log(f"stopping_early: wallclock_cap train_time: {training_time_ms:.0f}ms step: {step}/{h.iterations}")
            break
        elapsed_ms = training_time_ms + 1e3 * (time.perf_counter() - t0)
        frac = training_frac(step, elapsed_ms)
        scale = lr_mul(frac)
        if h.num_loops > 0 and not base_model.looping_active and frac >= h.enable_looping_at:
            base_model.looping_active = True
            log(
                f"layer_loop:enabled step:{step} frac:{frac:.3f} "
                f"encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}"
            )
        train_loss = step_fn(step, scale)
        with torch.no_grad():
            for name, tensor in base_model.state_dict().items():
                ema_state[name].mul_(ema_decay).add_(tensor.detach().float(), alpha=1.0 - ema_decay)
        step += 1
        approx_training_time_ms = training_time_ms + 1e3 * (time.perf_counter() - t0)
        should_log_train = h.train_log_every > 0 and (step <= 5 or step % h.train_log_every == 0 or stop_after_step is not None)
        if should_log_train:
            tok_per_sec = step * h.train_batch_tokens / (approx_training_time_ms / 1e3)
            log(
                f"{step}/{h.iterations} train_loss: {train_loss.item():.4f} "
                f"train_time: {approx_training_time_ms / 60000:.1f}m tok/s: {tok_per_sec:.0f}"
            )
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if h.distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log(
        "peak memory allocated: "
        f"{torch.cuda.max_memory_allocated() // 1024 // 1024} MiB reserved: "
        f"{torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )
    log("ema:applying EMA weights")
    current_state = base_model.state_dict()
    avg_state = {name: tensor.to(dtype=current_state[name].dtype) for name, tensor in ema_state.items()}
    base_model.load_state_dict(avg_state, strict=True)
    return base_model, compiled_model


def train_and_eval(h: Hyperparameters, device: torch.device) -> None:
    seed_all(h.seed)
    val_data = ValidationData(h, device)
    train_shards = len(list(Path(h.datasets_dir).resolve().glob("fineweb_train_*.bin")))
    log(f"train_shards: {train_shards}")
    log(f"val_tokens: {val_data.val_tokens.numel() - 1}")
    base_model, compiled_model = train_model(h, device, val_data)
    reset_dynamo()
    timed_eval("pre-quantization post-ema", eval_val, h, device, val_data, compiled_model)
    serialize(h, base_model, Path(__file__).read_text(encoding="utf-8"))
    if h.distributed:
        dist.barrier()
    eval_model = deserialize(h, device)
    if h.num_loops > 0:
        eval_model.looping_active = True
    compiled_eval_model = maybe_compile(eval_model, h.compile_enabled, dynamic=False, fullgraph=True)
    timed_eval("quantized", eval_val, h, device, val_data, compiled_eval_model)
    if h.sliding_window_enabled:
        timed_eval("quantized_sliding_window", eval_val_sliding, h, device, val_data, eval_model)
    if h.ttt_enabled and h.sliding_window_enabled:
        del eval_model, compiled_eval_model
        reset_dynamo()
        torch.cuda.empty_cache()
        ttt_model = deserialize(h, device)
        if h.num_loops > 0:
            ttt_model.looping_active = True
        timed_eval("quantized_ttt", eval_val_ttt, h, device, val_data, ttt_model)
        del ttt_model
    if h.etlb_enabled and h.sliding_window_enabled:
        if "eval_val_sliding_etlb" not in globals():
            raise RuntimeError("ETLB evaluation is not implemented in train_gpt_decoded.py")
        if "eval_model" not in locals():
            eval_model = deserialize(h, device)
            if h.num_loops > 0:
                eval_model.looping_active = True
        timed_eval("quantized_sliding_etlb", eval_val_sliding_etlb, h, device, val_data, eval_model)


def main() -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    from torch.backends.cuda import (
        enable_cudnn_sdp,
        enable_flash_sdp,
        enable_math_sdp,
        enable_mem_efficient_sdp,
    )

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)
    if hasattr(torch, "_dynamo"):
        torch._dynamo.config.optimize_ddp = False

    h = Hyperparameters()
    if h.lora_rank < 0:
        raise ValueError(f"LORA_RANK must be >= 0, got {h.lora_rank}")
    set_logging_hparams(h)
    if h.is_main_process:
        os.makedirs("logs", exist_ok=True)
        log("=" * 100, console=False)
        log("Hyperparameters:")
        for key, value in sorted(vars(type(h)).items()):
            if not key.startswith("_"):
                log(f"  {key}: {value}")
        log("=" * 100, console=False)
        log(f"Running Python {sys.version}", console=False)
        log(f"Running PyTorch {torch.__version__}", console=False)
        log(
            subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
            console=False,
        )
        log("=" * 100, console=False)
    train_and_eval(h, device)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
