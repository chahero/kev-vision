"""Experimental single-image Kev pooling using the original Qwen vision tower."""
from pathlib import Path
from types import SimpleNamespace

import torch
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.model_executor.models.interfaces_base import default_pooling_type
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration, Qwen3_5ProcessingInfo
from vllm.model_executor.models.qwen3_vl import Qwen3VLMultiModalProcessor, Qwen3VLDummyInputsBuilder

from .model import KevPointerPooler


@default_pooling_type(seq_pooling_type='LAST', tok_pooling_type='ALL')
@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor, info=Qwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class KevQwen3_5VisionForPooling(Qwen3_5ForConditionalGeneration):
    is_pooling_model = True

    def __init__(self, *, vllm_config, prefix=''):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        cfg=vllm_config.model_config.hf_config
        precision = getattr(cfg, 'kev_vision_precision', 'native')
        if precision not in ('native', 'reference'):
            raise ValueError(f'Unknown kev_vision_precision: {precision}')
        if precision == 'reference':
            if not vllm_config.model_config.enforce_eager:
                raise ValueError('Reference vision precision currently requires eager execution')
            from .precision import configure_reference_precision
            configure_reference_precision(self.visual, cfg.vision_config)
        head_cfg=SimpleNamespace(hidden_size=cfg.text_config.hidden_size,
                                 **{key:getattr(cfg,key) for key in ('kev_head_dim','kev_temperature','kev_option_end_id','kev_decide_id')})
        self.pooler=KevPointerPooler(head_cfg)
        self.head_path=Path(vllm_config.model_config.model)/'head.pt'

    def load_weights(self, weights):
        loaded=super().load_weights(weights)
        saved=torch.load(self.head_path,map_location='cpu',weights_only=True)
        for name,tensor in saved['head'].items():
            layer,attr=name.split('.')
            with torch.no_grad():
                getattr(getattr(self.pooler,layer),attr).copy_(tensor)
            loaded.add('pooler.'+name)
        return loaded
