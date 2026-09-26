# Experimental image-conditioned Kev decisions

Status: feasibility demonstrated on synthetic inputs; numerical parity is **not yet accepted**.
Branch: `experiment/vision`. Linux aarch64 / GB10 / BF16 only, 2026-09-27.
This is an unofficial extension, not an upstream Kev vision checkpoint or a production feature.

## What was implemented

`KevQwen3_5VisionForPooling` uses vLLM's Qwen3.5 multimodal backbone and the existing Kev pointer pooler. The original text class and default server commands remain text-only.

The reference experiment loads the exact full Qwen3.5-4B-Base revision, including its 297 vision tensors. It replaces only the language model's 426 tensors with the previously validated BF16 Kev merge, using strict key matching. No visual weights are borrowed from an Instruct model, and no parameters are trained. The original fp32 pointer head and its text-fitted temperature are reused. That temperature has not been calibrated for images.

The encoder inserts one image into the state, expands its image tokens, preserves the native multimodal positional encoding, and reads the pointer scores at the shifted option/decision positions. It counts image tokens toward the 1,024-token state and 2,048-token row budgets. User-supplied control tokens pass through the pinned Kev sanitizer.

The standalone reference and vLLM runs use separate processes. Existing GPU services were not stopped.

## Experiment and results

Fourteen deterministic 256×256 images cover three colors, three shapes, counts 1–3, three horizontal positions, and the printed words CAT/DOG. These are deliberately simple synthetic fixtures, not a representative visual benchmark. Filenames and expected labels are never model input. Text fixtures require the DejaVu Sans Bold system font; if absent, the script explicitly produces 12 cases instead of 14 (the actual count is recorded in the result).

For each question, three conditions are compared:

| Input condition | Transformers reference | vLLM default FlashAttention |
| --- | --- | --- |
| Correct image | 14/14 correct | 14/14 correct |
| White blank image | 5/14 correct | 5/14 correct |
| No image | 5/14 correct | 5/14 correct |

Within each category the question and choices are held fixed while the image changes. The change in selected answers demonstrates that visual information is used on these fixtures. Blank/no-image scores are controls, not desirable behavior or a missing-image detector.

All 42 image/control decisions and 27 existing text decisions matched their reference argmax. Text maximum probability error was 0.0080635548, within the existing 0.02 tolerance. Five direct Transformers full-model text checks also matched, with maximum error 0.0012509823. The existing eight Playground tests and three new image input checks passed.

**The probability parity test fails.** On the CAT image, the reference assigned CAT probability 0.6933595538; the default vLLM batch assigned 0.7234101295 (absolute difference 0.0300505757). This exceeds the predeclared 0.02 tolerance. The singleton CAT error with FlashAttention was 0.0349015594 (batch/singleton difference 0.0048509836). The tolerance has not been loosened.

A controlled run using `TORCH_SDPA` for vLLM's vision attention did not resolve this: the maximum batch error grew to 0.0690717697 and a singleton CAT request had error 0.0710175037. Its batch/singleton difference was only 0.0022567436. Processor pixel values for CAT matched exactly, and the enhanced vLLM test checks every expanded input-token sequence against the reference builder. The remaining difference has not been localized to a specific kernel or layer; attention-backend selection alone is not a fix.

[Reference results](vision-reference-result.json), [FlashAttention results](vision-vllm-flash-result.json), and [SDPA results](vision-vllm-sdpa-result.json) contain the full numerical evidence. The checked-in numerical evidence includes both successful decision matches and the failed tolerance gate. Do not describe this branch as passing full image parity, general OCR, real-photo classification, or image probability calibration.

## Reproduce on Linux

Use the project's pinned Linux environment. Starting from a prepared text workspace:

```bash
bash scripts/setup_kev_vllm.sh
# Creates the 27-row baseline in this workspace:
.venv/bin/python scripts/validate_kev_vllm.py
# Generates images, runs reference/control comparisons and exports full weights:
OMP_NUM_THREADS=4 .venv/bin/python scripts/experiment_vision.py --export
# Expected nonzero exit: the 0.02 probability gate is currently not satisfied.
HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 OMP_NUM_THREADS=4 \
  .venv/bin/python scripts/test_kev_vision_vllm.py --vision-attention FLASH_ATTN
# Diagnostic alternative, also expected to fail the same gate:
HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 OMP_NUM_THREADS=4 \
  .venv/bin/python scripts/test_kev_vision_vllm.py --vision-attention TORCH_SDPA
.venv/bin/python -m unittest discover -s tests -v
```

`experiment_vision.py --export` refuses to overwrite existing vision weights. To repeat reference scoring with the same inputs, omit `--export`. A new export needs its own archived/rebuilt output directory. Setup requires disk space for the existing text export plus approximately 8.46 GiB for the vision export. Image preprocessing configuration is fetched from the same pinned base revision.

Runtime files are under the ignored `artifacts/vision-experiment/`: generated PNGs, cases, reference and vLLM JSON results, model weights, and logs. On this machine the pre-existing pinned text model and reference records were copied from the earlier experiment; a completely fresh machine was not tested.

## Boundaries and next work

- Offline `vllm.LLM.encode` was exercised. The current HTTP query client, server launcher, and Playground do not accept images yet.
- Single image, fixed 256×256 fixtures, BF16, eager mode, no prefix caching/chunked prefill; no Windows validation.
- Real photos, document layouts, varied resolutions/aspect ratios, option permutations with images, robustness, and latency/memory benchmarks remain untested.
- Before merging a supported feature: localize the CAT numerical mismatch, measure realistic image-task accuracy and calibration, then add request validation and HTTP/UI image handling.
