"""CPU contract checks for the optional vision arithmetic configuration."""
import unittest
from types import SimpleNamespace

import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig

from kev_vision.precision import configure_reference_precision


class VisionPrecisionTests(unittest.TestCase):
    def setUp(self):
        self.config = Qwen3_5VisionConfig(hidden_size=32, num_heads=4,
                                         num_position_embeddings=16, spatial_merge_size=2)
        self.visual = torch.nn.Module()
        self.visual.pos_embed = torch.nn.Embedding(16, 32, dtype=torch.bfloat16)
        self.visual.device = torch.device('cpu')
        self.visual.dtype = torch.bfloat16
        self.visual.num_grid_per_side = 4
        self.visual.spatial_merge_size = 2
        self.visual.patch_embed = SimpleNamespace(proj=SimpleNamespace(enable_linear=True))
        self.visual.blocks = [SimpleNamespace(attn=SimpleNamespace(
            apply_rotary_emb=SimpleNamespace(enable_fp32_compute=False))) for _ in range(3)]

    def test_pinned_config_and_checkpoint_contract(self):
        before = {k:v.clone() for k,v in self.visual.state_dict().items()}
        configure_reference_precision(self.visual, self.config)
        self.assertFalse(self.visual.patch_embed.proj.enable_linear)
        self.assertTrue(all(b.attn.apply_rotary_emb.enable_fp32_compute for b in self.visual.blocks))
        self.assertEqual(set(before), set(self.visual.state_dict()))
        for key, value in self.visual.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))
        self.assertFalse(hasattr(self.config, 'head_dim'))
        cos, sin = self.visual.rot_pos_emb([[1, 2, 4]])
        self.assertEqual(cos.dtype, torch.float32)
        self.assertEqual(sin.dtype, torch.float32)
        self.assertEqual(tuple(cos.shape), (8, 4))
        self.assertTrue(torch.equal(cos[0], torch.ones(4)))
        self.assertTrue(torch.equal(sin[0], torch.zeros(4)))

    def test_mixed_grid_metadata_keeps_image_boundaries(self):
        configure_reference_precision(self.visual, self.config)
        grids = [[1, 2, 4], [1, 4, 2], [1, 4, 4]]
        combined = self.visual.fast_pos_embed_interpolate(grids)
        separate = torch.cat([self.visual.fast_pos_embed_interpolate([grid]) for grid in grids])
        self.assertTrue(torch.equal(combined, separate))
        cos, sin = self.visual.rot_pos_emb(grids)
        split = [self.visual.rot_pos_emb([grid]) for grid in grids]
        self.assertTrue(torch.equal(cos, torch.cat([pair[0] for pair in split])))
        self.assertTrue(torch.equal(sin, torch.cat([pair[1] for pair in split])))


if __name__ == '__main__':
    unittest.main()
