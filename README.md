# kev-vision

Experimental image-conditioned decisions using the Qwen3.5 vision tower, Kev language weights and pointer head, and vLLM pooling. No additional training; independent project, not an official Kev release.

[한국어](README.ko.md) · [Validation](docs/validation.md) · [Attribution](THIRD_PARTY.md)

## Run

Validated on Linux aarch64 / NVIDIA GB10, Python 3.12, BF16, eager mode. Install `uv` and a compatible NVIDIA driver, then run from the repository root:

```bash
bash scripts/setup_kev_vision.sh
# Generate the 27-row text reference and image reference.
OMP_NUM_THREADS=4 .venv/bin/python scripts/validate_kev_vllm.py --phase reference
OMP_NUM_THREADS=4 .venv/bin/python scripts/experiment_vision.py
.venv/bin/python -m unittest discover -s tests -v
HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 OMP_NUM_THREADS=4 \
  .venv/bin/python scripts/test_kev_vision_vllm.py \
  --vision-attention FLASH_ATTN --vision-precision reference --all-singletons
# Append --benchmark for cache-disabled regression and latency measurements.
```

Use `--vision-precision reference` for the validated arithmetic. Direct integrations pass `hf_overrides={"kev_vision_precision": "reference"}` to vLLM with eager execution. Omitting the option retains the native path, which fails the probability tolerance.

## Validation and scope

- 69 inputs × forward/reversed/singleton × cache on/off: 414 comparisons passed, matching reference argmax with maximum probability error 1.50 percentage points (tolerance: 2).
- Synthetic images: 14/14 correct; blank/no-image controls: 5/14 each. The 27 text decisions matched the reference.
- Local 256px singleton latency median: 61.5 ms versus native 60.6 ms. Excludes loading, input preparation and HTTP.

Real photos/documents, varied-resolution accuracy and image probability calibration remain unverified. No image HTTP API or UI is included. See the validation report for exact limits and results.

`artifacts/` contains locally downloaded inputs, exported weights and generated fixtures; `.venv/` is the local environment. Both are excluded from upload. Setup downloads pinned upstream inputs. No license for this project's original code has been selected; upstream components retain their own terms.

This project does not require the separate `kev-vllm` repository or package. Helper filenames retain `kev_vllm` from extraction, but their implementations are local. Setup downloads pinned upstream Kev encoder/checkpoint source and weights; the text merge is an input to the vision export.
