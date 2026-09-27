"""Choose among options for one local image using the validated vision profile."""
import argparse
import json
import sys
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True, help='Local image file')
    parser.add_argument('--question', required=True)
    parser.add_argument('--options', nargs='+', required=True)
    parser.add_argument('--state', default='Inspect the supplied image.')
    parser.add_argument('--output', type=Path, help='Save result JSON (console also contains engine logs)')
    args = parser.parse_args(argv)
    if not args.image.is_file():
        parser.error(f'Image not found: {args.image}')
    if not args.question.strip():
        parser.error('Question must not be empty')
    if len(args.options) < 2 or any(not option.strip() for option in args.options):
        parser.error('Supply at least two non-empty options')
    if len(set(args.options)) != len(args.options):
        parser.error('Options must be distinct')
    if args.output and args.output.resolve() == args.image.resolve():
        parser.error('Output must not overwrite the input image')
    return args


def main():
    args = parse_args()
    from experiment_vision import OUT, encode_inputs
    from kev_runtime import AUDIT, engine_options, WINDOWS
    from prepare_kev_vllm import load_pinned_kev

    model_path = OUT / 'merged-vision'
    for path in [model_path / name for name in ('config.json', 'model.safetensors', 'head.pt',
                                               'tokenizer.json', 'preprocessor_config.json')] + [AUDIT / 'kev-source/model.py', AUDIT / 'kev-source/checkpoint.py']:
        if not path.is_file():
            raise SystemExit(f'Missing {path}. Run: bash scripts/setup_kev_vision.sh')

    import torch
    from PIL import Image, ImageOps, UnidentifiedImageError
    from transformers import AutoTokenizer, AutoImageProcessor

    try:
        with Image.open(args.image) as source:
            # Preserve the whole image and aspect ratio within the validated square input.
            image = ImageOps.pad(ImageOps.exif_transpose(source).convert('RGB'),
                                 (256, 256), color='white')
    except (OSError, UnidentifiedImageError) as error:
        raise SystemExit(f'Cannot read image: {error}') from error
    load_pinned_kev()
    km = sys.modules['pinned_kev.model']
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
    case = {'state': args.state, 'question': args.question, 'options': args.options}
    try:
        expected_ids = encode_inputs(km, tokenizer, processor, case, image)['input_ids'][0].tolist()
        record = {'state': args.state, 'questions': [
            {'instr': args.question, 'options': args.options, 'label': 0}]}
        ids = km.encode(tokenizer, record, max_state=1024, max_branch=2048, strict=True)['ids']
    except km.ContextOverflow as error:
        raise SystemExit(f'Input too long: {error}. Shorten the state, question or options.') from error
    marker = [tokenizer.convert_tokens_to_ids(token) for token in
              ('<|vision_start|>', '<|image_pad|>', '<|vision_end|>')]
    prompt = {'prompt_token_ids': ids[:1] + marker + ids[1:], 'multi_modal_data': {'image': image}}

    from vllm import LLM, PoolingParams
    options = engine_options()
    options.update(model=str(model_path), limit_mm_per_prompt={'image': 1, 'video': 0},
                   mm_processor_kwargs={'min_pixels': 65536, 'max_pixels': 65536},
                   mm_encoder_attn_backend='TORCH_SDPA' if WINDOWS else 'FLASH_ATTN',
                   hf_overrides={'kev_vision_precision': 'reference'})
    model = LLM(**options)
    result = model.encode([prompt], pooling_task='token_embed',
                          pooling_params=PoolingParams(task='token_embed'), use_tqdm=False)[0]
    values = result.outputs.data.detach().cpu()
    if result.prompt_token_ids != expected_ids:
        raise RuntimeError('Image token expansion differs from the reference encoder')
    if tuple(values.shape) != (len(args.options), 1):
        raise RuntimeError(f'Unexpected result shape: {tuple(values.shape)}')
    values = values.flatten()
    if not (torch.isfinite(values).all() and (values >= 0).all() and
            (values <= 1).all() and abs(float(values.sum()) - 1) < 1e-5):
        raise RuntimeError('Invalid option probabilities')
    index = int(values.argmax())
    report = {'image': str(args.image), 'state': args.state, 'question': args.question,
              'selected_index': index, 'selected_option': args.options[index],
              'options': [{'text': option, 'probability': probability}
                          for option, probability in zip(args.options, values.tolist(), strict=True)],
              'vision_precision': 'reference', 'input_size': [256, 256]}
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding='utf-8')
    print(rendered, end='')


if __name__ == '__main__':
    main()
