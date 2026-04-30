# Copyright (c) 2026, Dao AI Lab, Goombalab.
# Memory module integration by user experiment.

import math
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated

try:
    from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo import mamba3_mimo as mamba3_mimo_combined
except ImportError:
    mamba3_mimo_combined = None

from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined

from mamba_ssm.ops.triton.mamba3.mamba3_mimo_rotary_step import apply_rotary_qk_inference_fwd

try:
    from mamba_ssm.ops.cute.mamba3.mamba3_step_fn import mamba3_step_fn
except ImportError:
    mamba3_step_fn = None


# ─────────────────────────────────────────────────────────────────────────────
# GatedMemoryModule
#
# Shapes used throughout (SISO mode; MIMO analogous with extra R dim):
#   u_seg          : (B, SegLen, DModel)
#   cached_pools   : list of (B, 1, DModel)       — mean-pool of each past segment
#   cached_states  : list of (B, NHeads, HeadDim, DState)  — final SSM state per segment
#
# Retrieval output: (B, SegLen, NHeads, HeadDim)
# ─────────────────────────────────────────────────────────────────────────────
class GatedMemoryModule(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int,
        nheads: int,
        headdim: int,
        max_segments: int = 16,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.nheads = nheads
        self.headdim = headdim
        self.max_segments = max_segments

        # Projects current segment hidden states into a query for memory lookup.
        # Kept as d_model→d_model so pool comparisons are in the same space.
        self.gate_proj = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)

        # Per-head projection from DState → 1 to collapse the state dimension
        # after the weighted sum.  Shape: (NHeads, DState).
        self.state_proj = nn.Parameter(
            torch.ones(nheads, d_state, **factory_kwargs) / d_state
        )

        # Learned scalar gate that controls how much retrieved memory contributes.
        # Initialised near 0 so the model starts ignoring memory and learns to use it.
        self.blend_gate = nn.Parameter(torch.zeros(1, **factory_kwargs))

    # -------------------------------------------------------------------------
    def forward(
        self,
        u_seg: torch.Tensor,                  # (B, SegLen, DModel)
        cached_states: list,                  # list of (B, NHeads, HeadDim, DState)
        cached_pools: list,                   # list of (B, 1, DModel)
    ) -> torch.Tensor | None:
        """
        Returns (B, SegLen, NHeads, HeadDim) memory contribution, or None if
        there is no cached history yet.
        """
        if not cached_states:
            return None

        batch, seg_len, _ = u_seg.shape
        num_past = len(cached_states)

        # ── 1. Compute gating query from current segment ─────────────────────
        # u_gate: (B, SegLen, DModel)
        u_gate = self.gate_proj(u_seg)

        # pools: (B, NumPast, DModel)
        pools = torch.cat(cached_pools, dim=1)

        # ── 2. Scaled dot-product attention over past segments ────────────────
        # scores: (B, SegLen, NumPast)
        scale = self.d_model ** 0.5
        scores = torch.einsum("bsd,bnd->bsn", u_gate, pools) / scale
        gamma = torch.softmax(scores, dim=-1)          # (B, SegLen, NumPast)

        # ── 3. Weighted sum of past SSM states ────────────────────────────────
        # states_stack: (B, NumPast, NHeads, HeadDim, DState)
        states_stack = torch.stack(cached_states, dim=1).to(u_seg.dtype)

        # gated_state: (B, SegLen, NHeads, HeadDim, DState)
        gated_state = torch.einsum("bsn,bnhpd->bshpd", gamma, states_stack)

        # ── 4. Project DState → scalar per head-dim position ─────────────────
        # state_proj: (NHeads, DState) — broadcast over B, SegLen, HeadDim
        # y_mem: (B, SegLen, NHeads, HeadDim)
        y_mem = torch.einsum("bshpd,hd->bshp", gated_state, self.state_proj)

        return y_mem

    # -------------------------------------------------------------------------
    @staticmethod
    def update_cache(
        cached_states: list,
        cached_pools: list,
        ssm_state: torch.Tensor,   # (B, NHeads, HeadDim, DState)
        u_seg: torch.Tensor,       # (B, SegLen, DModel)
        max_segments: int,
    ):
        """
        Functional update — returns new lists, does not mutate inputs.
        States are detached: gradients flow through the online path only.
        Change .detach() to nothing if you want end-to-end gradient flow
        through memory (more expressive but less stable to train).
        """
        pool = u_seg.mean(dim=1, keepdim=True).detach()   # (B, 1, DModel)
        new_states = cached_states + [ssm_state.detach()]
        new_pools  = cached_pools  + [pool]

        # Sliding window — drop oldest segments beyond budget
        if len(new_states) > max_segments:
            new_states = new_states[-max_segments:]
            new_pools  = new_pools[-max_segments:]

        return new_states, new_pools


# ─────────────────────────────────────────────────────────────────────────────
# Mamba3  (original + optional segmented memory)
# ─────────────────────────────────────────────────────────────────────────────
class Mamba3(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=128,
        expand=2,
        headdim=64,
        ngroups=1,
        # ── Mamba-3 configs ───────────────────────────────────────────────────
        rope_fraction=0.5,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        A_floor=1e-4,
        is_outproj_norm=False,
        is_mimo=False,
        mimo_rank=4,
        # ── Memory module configs ─────────────────────────────────────────────
        use_segmented_memory=False,   # set True to enable the memory module
        segment_size=512,             # tokens per segment
        max_memory_segments=16,       # sliding-window cap on cached segments
        # ── Fused kernel and sharding options ─────────────────────────────────
        chunk_size=64,
        dropout=0.0,
        layer_idx=None,
        n_layer=None,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.headdim = headdim
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx
        self.A_floor = A_floor
        self.is_outproj_norm = is_outproj_norm
        self.is_mimo = is_mimo
        self.mimo_rank = mimo_rank
        if not self.is_mimo:
            self.mimo_rank = 1
        else:
            assert mamba3_mimo_combined is not None, (
                "Fails to import Mamba-3 MIMO kernels. "
                "Please ensure you installed TileLang."
            )

        # ── Memory module ─────────────────────────────────────────────────────
        self.use_segmented_memory = use_segmented_memory
        self.segment_size = segment_size
        if use_segmented_memory:
            # Defer nheads computation; we build memory_module after nheads is known.
            # Handled below after self.nheads is set.
            self._memory_max_segments = max_memory_segments

        self.d_inner = int(self.expand * self.d_model)
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.num_bc_heads = ngroups

        # RoPE flags
        assert rope_fraction in [0.5, 1.0]
        self.rotary_dim_divisor = int(2 / rope_fraction)
        self.split_tensor_size = int(d_state * rope_fraction)
        if self.split_tensor_size % 2 != 0:
            self.split_tensor_size -= 1
        self.num_rope_angles = self.split_tensor_size // 2
        assert self.num_rope_angles > 0

        # Order: [z, x, B, C, dd_dt, dd_A, trap, angle]
        d_in_proj = (
            2 * self.d_inner
            + 2 * self.d_state * self.num_bc_heads * self.mimo_rank
            + 3 * self.nheads
            + self.num_rope_angles
        )
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=False, **factory_kwargs)

        # dt_bias parameterisation
        _dt = torch.exp(
            torch.rand(self.nheads, device=device, dtype=torch.float32)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        _dt = torch.clamp(_dt, min=dt_init_floor)
        _dt_bias = _dt + torch.log(-torch.expm1(-_dt))
        self.dt_bias = nn.Parameter(_dt_bias, requires_grad=True)
        self.dt_bias._no_weight_decay = True

        # B and C biases
        self.B_bias = nn.Parameter(
            1 + torch.zeros((self.nheads, self.mimo_rank, self.d_state),
                            dtype=torch.float32, device=device),
            requires_grad=True,
        )
        self.C_bias = nn.Parameter(
            1 + torch.zeros((self.nheads, self.mimo_rank, self.d_state),
                            dtype=torch.float32, device=device),
            requires_grad=True,
        )

        # RMS Norm for B and C
        assert RMSNormGated is not None
        self.B_norm = RMSNormGated(self.d_state, eps=1e-5, **factory_kwargs)
        self.C_norm = RMSNormGated(self.d_state, eps=1e-5, **factory_kwargs)

        if self.is_mimo:
            mimo_x_init = torch.ones(self.nheads, self.mimo_rank, self.headdim, device=device) / self.mimo_rank
            mimo_z_init = torch.ones(self.nheads, self.mimo_rank, self.headdim, device=device)
            mimo_o_init = torch.ones(self.nheads, self.mimo_rank, self.headdim, device=device) / self.mimo_rank
            self.mimo_x = nn.Parameter(mimo_x_init, requires_grad=True)
            self.mimo_z = nn.Parameter(mimo_z_init, requires_grad=True)
            self.mimo_o = nn.Parameter(mimo_o_init, requires_grad=True)

        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        if self.is_outproj_norm:
            self.norm = RMSNormGated(
                self.d_inner,
                eps=1e-5,
                norm_before_gate=True,
                group_size=self.headdim,
                **factory_kwargs,
            )

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False, **factory_kwargs)

        # Build memory module now that self.nheads is known
        if use_segmented_memory:
            self.memory_module = GatedMemoryModule(
                d_model=self.d_model,
                d_state=self.d_state,
                nheads=self.nheads,
                headdim=self.headdim,
                max_segments=self._memory_max_segments,
                device=device,
                dtype=dtype,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers shared by full-sequence and segmented paths
    # ─────────────────────────────────────────────────────────────────────────

    def _project_inputs(self, u_chunk):
        """
        Run in_proj + split + rearrange for a chunk u_chunk: (B, L, DModel).
        Returns z, x, B, C, DT, ADT, trap, angles — all in kernel-ready shapes.
        """
        zxBCdtAtrap = self.in_proj(u_chunk)
        z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
            zxBCdtAtrap,
            [
                self.d_inner, self.d_inner,
                self.d_state * self.num_bc_heads * self.mimo_rank,
                self.d_state * self.num_bc_heads * self.mimo_rank,
                self.nheads, self.nheads, self.nheads,
                self.num_rope_angles,
            ],
            dim=-1,
        )
        z = rearrange(z, "b l (h p) -> b l h p", p=self.headdim)
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        B = rearrange(B, "b l (r g n) -> b l r g n", r=self.mimo_rank, g=self.num_bc_heads)
        C = rearrange(C, "b l (r g n) -> b l r g n", r=self.mimo_rank, g=self.num_bc_heads)
        trap = rearrange(trap, "b l h -> b h l")

        _A = -F.softplus(dd_A.to(torch.float32))
        _A = torch.clamp(_A, max=-self.A_floor)
        DT  = F.softplus(dd_dt + self.dt_bias)          # (B, L, NHeads)
        ADT = _A * DT
        DT  = rearrange(DT,  "b l n -> b n l")
        ADT = rearrange(ADT, "b l n -> b n l")

        angles = angles.unsqueeze(-2).expand(-1, -1, self.nheads, -1).to(torch.float32)

        B = self.B_norm(B)
        C = self.C_norm(C)

        return z, x, B, C, DT, ADT, trap, angles

    def _run_siso_kernel(self, z, x, B, C, DT, ADT, trap, angles,
                         input_states=None, return_final_states=False,
                         cu_seqlens=None):
        """
        Thin wrapper around mamba3_siso_combined.
        input_states: tuple (angle, ssm, k, v) or None
        Returns y: (B, L, NHeads, HeadDim), plus state tuple if requested.
        """
        # Unpack optional input states for warm-starting a segment
        init_states = None
        if input_states is not None:
            init_states = input_states  # passed as Input_States kwarg if supported

        result = mamba3_siso_combined(
            Q=C.squeeze(2),
            K=B.squeeze(2),
            V=x,
            ADT=ADT,
            DT=DT,
            Trap=trap,
            Q_bias=self.C_bias.squeeze(1),
            K_bias=self.B_bias.squeeze(1),
            Angles=angles,
            D=self.D,
            Z=z if not self.is_outproj_norm else None,
            chunk_size=self.chunk_size,
            Input_States=init_states,
            return_final_states=return_final_states,
            cu_seqlens=cu_seqlens,
        )

        if return_final_states:
            y, last_angle, last_ssm, last_k, last_v, *_ = result
            if self.is_outproj_norm:
                z_flat = rearrange(z, "b l h p -> b l (h p)")
                y_flat = rearrange(y, "b l h p -> b l (h p)")
                y_flat = self.norm(y_flat, z_flat)
                y = rearrange(y_flat, "b l (h p) -> b l h p", p=self.headdim)
            return y, (last_angle, last_ssm, last_k, last_v)
        else:
            y = result
            if self.is_outproj_norm:
                z_flat = rearrange(z, "b l h p -> b l (h p)")
                y_flat = rearrange(y, "b l h p -> b l (h p)")
                y_flat = self.norm(y_flat, z_flat)
                y = rearrange(y_flat, "b l (h p) -> b l h p", p=self.headdim)
            return y, None

    def _run_mimo_kernel(self, z, x, B, C, DT, ADT, trap, angles,
                         return_final_states=False, cu_seqlens=None):
        """
        Thin wrapper around mamba3_mimo_combined.
        Returns y: (B, L, NHeads, HeadDim), plus state tuple if requested.
        """
        result = mamba3_mimo_combined(
            Q=C,
            K=B,
            V=x,
            ADT=ADT,
            DT=DT,
            Trap=trap,
            Q_bias=self.C_bias,
            K_bias=self.B_bias,
            MIMO_V=self.mimo_x,
            MIMO_Z=self.mimo_z,
            MIMO_Out=self.mimo_o if not self.is_outproj_norm else None,
            Angles=angles,
            D=self.D,
            Z=z if not self.is_outproj_norm else None,
            chunk_size=self.chunk_size,
            rotary_dim_divisor=self.rotary_dim_divisor,
            dtype=x.dtype,
            return_state=return_final_states,
            cu_seqlens=cu_seqlens,
        )

        if return_final_states:
            y, last_angle, last_ssm, last_k, last_v, *_ = result
            if self.is_outproj_norm:
                z_r = torch.einsum("blhp,hrp->blrhp", z.float(), self.mimo_z)
                z_r = rearrange(z_r, "b l r h p -> b l r (h p)")
                y   = rearrange(y,   "b l r h p -> b l r (h p)").float()
                y   = self.norm(y, z_r)
                y   = rearrange(y, "b l r (h p) -> b l r h p", p=self.headdim)
                y   = torch.einsum("blrhp,hrp->blhp", y, self.mimo_o)
            else:
                y = rearrange(y, "b l h p -> b l h p")  # already correct shape
            return y, (last_angle, last_ssm, last_k, last_v)
        else:
            if self.is_outproj_norm:
                z_r = torch.einsum("blhp,hrp->blrhp", z.float(), self.mimo_z)
                z_r = rearrange(z_r, "b l r h p -> b l r (h p)")
                result = rearrange(result, "b l r h p -> b l r (h p)").float()
                result = self.norm(result, z_r)
                result = rearrange(result, "b l r (h p) -> b l r h p", p=self.headdim)
                result = torch.einsum("blrhp,hrp->blhp", result, self.mimo_o)
            y = rearrange(result, "b l h p -> b l h p")
            return y, None

    # ─────────────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, u, seq_idx=None, cu_seqlens=None, inference_params=None):
        """
        u: (batch, seqlen, hidden_dim)
        Returns: same shape as u
        """
        batch, seqlen, dim = u.shape

        # ── Inference (decode) path — unchanged ──────────────────────────────
        angle_dt_state = ssm_state = k_state = v_state = None
        if inference_params is not None:
            inference_batch = (
                cu_seqlens.shape[0] - 1 if cu_seqlens is not None else batch
            )
            angle_dt_state, ssm_state, k_state, v_state = self._get_states_from_cache(
                inference_params, inference_batch
            )
            if inference_params.seqlen_offset > 0:
                out, _, _, _, _ = self.step(u, angle_dt_state, ssm_state, k_state, v_state)
                return out

        # ── Segmented memory path (training / prefill only) ──────────────────
        if self.use_segmented_memory:
            return self._forward_segmented(u, cu_seqlens, ssm_state)

        # ── Original full-sequence path ───────────────────────────────────────
        return self._forward_full(
            u, cu_seqlens, angle_dt_state, ssm_state, k_state, v_state
        )

    # ─────────────────────────────────────────────────────────────────────────

    def _forward_full(self, u, cu_seqlens, angle_dt_state, ssm_state, k_state, v_state):
        """Original single-pass logic, refactored into a helper."""
        z, x, B, C, DT, ADT, trap, angles = self._project_inputs(u)
        need_states = ssm_state is not None

        if self.is_mimo:
            y, states = self._run_mimo_kernel(
                z, x, B, C, DT, ADT, trap, angles,
                return_final_states=need_states,
                cu_seqlens=cu_seqlens,
            )
        else:
            y, states = self._run_siso_kernel(
                z, x, B, C, DT, ADT, trap, angles,
                return_final_states=need_states,
                cu_seqlens=cu_seqlens,
            )

        if need_states:
            last_angle, last_ssm, last_k, last_v = states
            angle_dt_state.copy_(last_angle)
            ssm_state.copy_(last_ssm)
            k_state.copy_(last_k if self.is_mimo else last_k.unsqueeze(1))
            v_state.copy_(last_v)

        y_flat = rearrange(y, "b l h p -> b l (h p)")
        return self.out_proj(y_flat.to(u.dtype))

    # ─────────────────────────────────────────────────────────────────────────

    def _forward_segmented(self, u, cu_seqlens, prefill_ssm_state=None):
        """
        Process u in non-overlapping segments of self.segment_size tokens.
        After each segment, the final SSM state is stored in the memory module's
        rolling cache and retrieved for subsequent segments via gated attention.

        Memory contribution is added to the per-head output before out_proj.

        Shape bookkeeping:
          y_seg   : (B, SegLen, NHeads, HeadDim)
          y_mem   : (B, SegLen, NHeads, HeadDim)  — from GatedMemoryModule
          blend   : scalar ∈ (0, 1)               — sigmoid of learned blend_gate
        """
        batch, seqlen, _ = u.shape
        output_list = []
        cached_states: list = []
        cached_pools:  list = []

        for start in range(0, seqlen, self.segment_size):
            end   = min(start + self.segment_size, seqlen)
            u_seg = u[:, start:end, :]   # (B, SegLen, DModel)

            # Project and run kernel for this segment, always requesting final states
            z, x, B, C, DT, ADT, trap, angles = self._project_inputs(u_seg)

            if self.is_mimo:
                y_seg, states = self._run_mimo_kernel(
                    z, x, B, C, DT, ADT, trap, angles,
                    return_final_states=True,
                    cu_seqlens=cu_seqlens,
                )
            else:
                y_seg, states = self._run_siso_kernel(
                    z, x, B, C, DT, ADT, trap, angles,
                    return_final_states=True,
                    cu_seqlens=cu_seqlens,
                )

            # states: (last_angle, last_ssm, last_k, last_v)
            # last_ssm: (B, NHeads, HeadDim, DState) — this is what we cache
            _, last_ssm, _, _ = states

            # ── Memory retrieval ──────────────────────────────────────────────
            y_mem = self.memory_module(u_seg, cached_states, cached_pools)
            # y_mem: (B, SegLen, NHeads, HeadDim) or None

            if y_mem is not None:
                # Learned blend gate: initialised ~0 so model starts ignoring memory
                blend = torch.sigmoid(self.memory_module.blend_gate)
                y_seg = y_seg + blend * y_mem

            output_list.append(y_seg)

            # ── Update rolling cache (functional, no mutation) ────────────────
            cached_states, cached_pools = GatedMemoryModule.update_cache(
                cached_states,
                cached_pools,
                last_ssm,                               # (B, NHeads, HeadDim, DState)
                u_seg,                                  # (B, SegLen, DModel)
                self.memory_module.max_segments,
            )

        # Concatenate segments and project to output
        # y_full: (B, SeqLen, NHeads, HeadDim)
        y_full = torch.cat(output_list, dim=1)
        y_flat = rearrange(y_full, "b l h p -> b l (h p)")
        return self.out_proj(y_flat.to(u.dtype))

    # ─────────────────────────────────────────────────────────────────────────
    # Step (decode) — unchanged from original
    # ─────────────────────────────────────────────────────────────────────────

    def _preprocess(self, A_proj, dd_dt, B, C, x, z, trap_proj, angle_proj):
        _A = -F.softplus(A_proj.to(torch.float32))
        _A = torch.clamp(_A, max=-self.A_floor)
        DT = F.softplus(dd_dt + self.dt_bias)
        trap = torch.sigmoid(trap_proj)

        rank = self.mimo_rank if self.is_mimo else 1
        B = rearrange(B, "b (r g s) -> b r g s", g=self.num_bc_heads, r=rank)
        C = rearrange(C, "b (r g s) -> b r g s", g=self.num_bc_heads, r=rank)

        B = self.B_norm(B)
        C = self.C_norm(C)

        B = B.expand(-1, -1, self.nheads, -1)
        C = C.expand(-1, -1, self.nheads, -1)

        x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
        z = rearrange(z, "b (h p) -> b h p", p=self.headdim)

        angles = angle_proj.unsqueeze(-2).expand(-1, self.nheads, -1)

        return DT, B, C, x, z, trap, _A, angles

    def _postprocess(self, y, outpj, z, zpj, headdim):
        z_r = torch.einsum("bhp,rhp->brhp", z.float(), zpj)
        z_r = rearrange(z_r, "b r h p -> b r (h p)")
        y   = rearrange(y,   "b r h p -> b r (h p)").float()
        y   = self.norm(y, z_r)
        y   = rearrange(y, "b r (h p) -> b r h p", p=headdim)
        y   = torch.einsum("brhp,rhp->bhp", y, outpj)
        return y

    def step(self, u, angle_state, ssm_state, k_state, v_state, **kwargs):
        assert mamba3_step_fn is not None, (
            "Cute Mamba-3 step function is not available. "
            "Please ensure you installed nvidia-cutlass-dsl and quack-kernels."
        )

        zxBCdt = self.in_proj(u)
        z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
            zxBCdt,
            [
                self.d_inner, self.d_inner,
                self.d_state * self.num_bc_heads * self.mimo_rank,
                self.d_state * self.num_bc_heads * self.mimo_rank,
                self.nheads, self.nheads, self.nheads,
                self.num_rope_angles,
            ],
            dim=-1,
        )

        DT, B, C, x, z, trap, A, angles = self._preprocess(
            dd_A, dd_dt, B, C, x, z, trap, angles
        )

        bias_q = rearrange(self.C_bias, "h r n -> r h n")
        bias_k = rearrange(self.B_bias, "h r n -> r h n")

        rotate_pairwise = not self.is_mimo
        C, B, nxt_angle_state = apply_rotary_qk_inference_fwd(
            q=C, k=B, angle_state=angle_state,
            angle_proj=angles, dt=DT, bias_q=bias_q, bias_k=bias_k,
            conjugate=False, inplace=False,
            rotate_pairwise=rotate_pairwise,
        )

        nxt_v_state = x
        nxt_k_state = B

        if self.is_mimo:
            xpj   = rearrange(self.mimo_x, "h r p -> r h p", p=self.headdim).contiguous()
            zpj   = rearrange(self.mimo_z, "h r p -> r h p", p=self.headdim).contiguous()
            outpj = rearrange(self.mimo_o, "h r p -> r h p", p=self.headdim).contiguous()
        else:
            xpj   = torch.ones(self.mimo_rank, self.nheads, self.headdim, device=x.device, dtype=x.dtype)
            zpj   = torch.ones(self.mimo_rank, self.nheads, self.headdim, device=z.device, dtype=z.dtype)
            outpj = torch.ones(self.mimo_rank, self.nheads, self.headdim, device=x.device, dtype=x.dtype)

        if self.is_outproj_norm:
            batch = x.shape[0]
            y = torch.empty(batch, self.mimo_rank, self.nheads, self.headdim,
                            device=x.device, dtype=x.dtype)
            mamba3_step_fn(
                ssm_state, k_state, v_state, A, B, C, self.D, x, DT, trap, xpj,
                outproj=None, state_out=None, out=y, z=None, zproj=None,
                tile_D=64, num_warps=4,
            )
            y = self._postprocess(y, outpj, z, zpj, self.headdim)
        else:
            y = torch.empty_like(x)
            mamba3_step_fn(
                ssm_state, k_state, v_state, A, B, C, self.D, x, DT, trap, xpj,
                outproj=outpj, state_out=None, out=y, z=z, zproj=zpj,
                tile_D=64, num_warps=4,
            )

        out = rearrange(y, "b h p -> b (h p)")
        out = self.out_proj(out.to(x.dtype))

        angle_state.copy_(nxt_angle_state)
        k_state.copy_(nxt_k_state)
        v_state.copy_(nxt_v_state)

        return out, nxt_angle_state, ssm_state, nxt_k_state, nxt_v_state

    # ─────────────────────────────────────────────────────────────────────────
    # Cache helpers — unchanged from original
    # ─────────────────────────────────────────────────────────────────────────

    def allocate_inference_cache(self, batch_size, max_seqlen,
                                 device=None, dtype=None, **kwargs):
        device = self.in_proj.weight.device if device is None else device
        dtype  = self.in_proj.weight.dtype  if dtype  is None else dtype

        angle_dt_state = torch.zeros(
            (batch_size, self.nheads, self.num_rope_angles),
            device=device, dtype=torch.float32,
        )
        ssm_state = torch.zeros(
            (batch_size, self.nheads, self.headdim, self.d_state),
            device=device, dtype=torch.float32,
        )
        k_shape = (
            (batch_size, self.mimo_rank, self.nheads, self.d_state)
            if self.is_mimo
            else (batch_size, 1, self.nheads, self.d_state)
        )
        k_state = torch.zeros(k_shape, device=device, dtype=dtype)
        v_state = torch.zeros(
            (batch_size, self.nheads, self.headdim), device=device, dtype=dtype
        )
        return (angle_dt_state, ssm_state, k_state, v_state)

    def _get_states_from_cache(self, inference_params, batch_size,
                               initialize_states=False):
        assert self.layer_idx is not None
        device = self.in_proj.weight.device
        dtype  = self.in_proj.weight.dtype

        if self.layer_idx not in inference_params.key_value_memory_dict:
            angle_dt_state = torch.zeros(
                (batch_size, self.nheads, self.num_rope_angles),
                device=device, dtype=torch.float32,
            )
            ssm_state = torch.zeros(
                (batch_size, self.nheads, self.headdim, self.d_state),
                device=device, dtype=torch.float32,
            )
            k_shape = (
                (batch_size, self.mimo_rank, self.nheads, self.d_state)
                if self.is_mimo
                else (batch_size, 1, self.nheads, self.d_state)
            )
            k_state = torch.zeros(k_shape, device=device, dtype=dtype)
            v_state = torch.zeros(
                (batch_size, self.nheads, self.headdim), device=device, dtype=dtype
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (
                angle_dt_state, ssm_state, k_state, v_state
            )
        else:
            angle_dt_state, ssm_state, k_state, v_state = (
                inference_params.key_value_memory_dict[self.layer_idx]
            )
            if initialize_states:
                angle_dt_state.zero_()
                ssm_state.zero_()
                k_state.zero_()
                v_state.zero_()

        return angle_dt_state, ssm_state, k_state, v_state