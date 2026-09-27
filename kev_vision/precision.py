"""Opt-in reference arithmetic for the pinned Qwen3.5 vision encoder.

Computes from the current image grid and checkpoint weights. No diagnostic
captures, fixture names, reference answers, or reference hidden states are read.
Attention and language-model kernels remain native vLLM implementations.
"""
from types import MethodType
from copy import deepcopy

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionRotaryEmbedding
from transformers.vision_utils import (
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
)


def _reference_position(self, grid_thw):
    grid = torch.as_tensor(grid_thw, dtype=torch.long, device=self.device)
    indices, weights = get_vision_interpolation_indices_and_weights(
        grid, self.num_grid_per_side, mode='bilinear', align_corners=True,
        spatial_merge_size=self.spatial_merge_size)
    # Retain FP32 interpolation weights/products until the sum is complete.
    return (self.pos_embed(indices) * weights[:, :, None]).sum(1).to(self.dtype)


def _reference_rotary(self, grid_thw):
    grid = torch.as_tensor(grid_thw, dtype=torch.long, device=self.device)
    positions = get_vision_position_ids(grid, self.spatial_merge_size)
    cos, sin = self.reference_rotary(self.pos_embed.weight, positions)
    # Native vLLM rotation consumes one half of the duplicated Neox table.
    half = cos.shape[-1] // 2
    return cos[:, :half].contiguous(), sin[:, :half].contiguous()


def configure_reference_precision(visual, vision_config):
    """Configure a newly created eager encoder before any warmup/metadata cache."""
    visual.patch_embed.proj.enable_linear = False
    visual.fast_pos_embed_interpolate = MethodType(_reference_position, visual)
    # Recompute FP32 frequencies, rather than upcasting a rounded BF16 cache.
    # The helper has no trainable parameters or persistent checkpoint buffers.
    rotary_config = deepcopy(vision_config)
    # The standalone vLLM config has num_heads, whereas the HF model loader
    # normally supplies the normalized head_dim before constructing RoPE.
    rotary_config.head_dim = vision_config.hidden_size // vision_config.num_heads
    with torch.device('cpu'):
        visual.reference_rotary = Qwen3_5VisionRotaryEmbedding(rotary_config)
    visual.reference_rotary.to(device=visual.device)
    visual.rot_pos_emb = MethodType(_reference_rotary, visual)
    for block in visual.blocks:
        block.attn.apply_rotary_emb.enable_fp32_compute = True
