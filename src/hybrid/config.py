from dataclasses import dataclass


@dataclass
class MixerConfig:
    d_model: int = 256
    # attention
    attn_head_dim: int = 64  # must equal ssm_d_state (Unified RoPE, A2)
    attn_window: int = 128
    # ssm (Mamba-2 / SSD)
    ssm_expand: int = 2
    ssm_head_dim: int = 64
    ssm_d_state: int = 64
    ssm_conv_kernel: int = 4
    ssm_chunk: int = 64
    # rope
    rope_base: float = 10000.0
    # engram
    vocab_size: int = 50257
    engram_orders: tuple[int, ...] = (2, 3)
    engram_heads_per_order: int = 4
    engram_head_dim: int = 32
    engram_table_size: int = 10007  # first prime; subsequent tables use the next distinct primes
    engram_conv_kernel: int = 4
    engram_seed: int = 1234

    def __post_init__(self):
        if self.d_model % self.attn_head_dim:
            raise ValueError(f"d_model={self.d_model} not divisible by attn_head_dim={self.attn_head_dim}")
        if (self.ssm_expand * self.d_model) % self.ssm_head_dim:
            raise ValueError(f"ssm_expand*d_model={self.ssm_expand * self.d_model} not divisible by ssm_head_dim={self.ssm_head_dim}")
        if self.attn_head_dim != self.ssm_d_state:
            raise ValueError(f"attn_head_dim={self.attn_head_dim} != ssm_d_state={self.ssm_d_state}; Unified RoPE needs one rotation (A2)")
        # attn_head_dim == ssm_d_state is checked above, so this also covers the SSM RoPE dim
        if self.attn_head_dim % 2:
            raise ValueError(f"RoPE dim attn_head_dim={self.attn_head_dim} must be even")
        if any(n < 2 for n in self.engram_orders):
            raise ValueError(f"engram_orders={self.engram_orders} must all be >= 2")
