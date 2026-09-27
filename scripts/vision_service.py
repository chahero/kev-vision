"""Persistent local image decisions; shares the CLI's validated inference path."""
import sys
import time
from pathlib import Path

from kev_runtime import AUDIT, WINDOWS, engine_options
from experiment_vision import OUT, encode_inputs
from prepare_kev_vllm import load_pinned_kev


class VisionEngine:
    def __init__(self):
        import torch
        from transformers import AutoTokenizer, AutoImageProcessor
        from vllm import LLM, PoolingParams
        model_path = OUT / 'merged-vision'
        for name in ('config.json', 'model.safetensors', 'head.pt', 'tokenizer.json', 'preprocessor_config.json'):
            if not (model_path / name).is_file():
                raise FileNotFoundError(f'Missing {model_path / name}; run the setup script first.')
        load_pinned_kev()
        self.encoder = sys.modules['pinned_kev.model']
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
        self.torch = torch
        self.params = PoolingParams(task='token_embed')
        options = engine_options()
        options.update(model=str(model_path), limit_mm_per_prompt={'image': 1, 'video': 0},
                       mm_processor_kwargs={'min_pixels': 65536, 'max_pixels': 65536},
                       mm_encoder_attn_backend='TORCH_SDPA' if WINDOWS else 'FLASH_ATTN',
                       hf_overrides={'kev_vision_precision': 'reference'})
        self.model = LLM(**options)

    def decide(self, image, state, question, options):
        started = time.perf_counter()
        from PIL import ImageOps
        image = ImageOps.pad(ImageOps.exif_transpose(image).convert('RGB'), (256, 256), color='white')
        case = dict(state=state, question=question, options=options)
        expected = encode_inputs(self.encoder, self.tokenizer, self.processor, case, image)['input_ids'][0].tolist()
        record = {'state': state, 'questions': [{'instr': question, 'options': options, 'label': 0}]}
        ids = self.encoder.encode(self.tokenizer, record, max_state=1024, max_branch=2048, strict=True)['ids']
        marker = [self.tokenizer.convert_tokens_to_ids(t) for t in ('<|vision_start|>', '<|image_pad|>', '<|vision_end|>')]
        prompt = {'prompt_token_ids': ids[:1] + marker + ids[1:], 'multi_modal_data': {'image': image}}
        result = self.model.encode([prompt], pooling_task='token_embed', pooling_params=self.params, use_tqdm=False)[0]
        values = result.outputs.data.detach().cpu()
        if result.prompt_token_ids != expected or tuple(values.shape) != (len(options), 1):
            raise RuntimeError('Model output does not match the encoded input')
        values = values.flatten()
        if not (self.torch.isfinite(values).all() and (values >= 0).all() and (values <= 1).all() and abs(float(values.sum()) - 1) < 1e-5):
            raise RuntimeError('Invalid model probabilities')
        selected = int(values.argmax())
        return dict(selected=options[selected], selected_index=selected,
                    probabilities=[dict(option=o, probability=p) for o, p in zip(options, values.tolist(), strict=True)],
                    tokens=len(expected), elapsed_ms=round((time.perf_counter()-started)*1000, 1),
                    input_size=[256, 256], vision_precision='reference')
