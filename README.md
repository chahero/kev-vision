# kev-vision

Experimental image-conditioned decisions using the original Qwen3.5 vision tower, Kev language adaptation and pointer head, and vLLM pooling. Extracted from the local `kev-vllm` vision experiment; this is an independent research project and is not an official Kev release.

[한국어](README.ko.md) · [Experiment details](docs/vision-experiment.md)

## Current evidence

- No additional training. All 14 simple synthetic images were classified correctly by both Transformers and vLLM. Blank/no-image controls each scored 5/14.
- All 27 existing text decisions matched the reference.
- **Probability parity remains failed**: the CAT fixture differs by 0.03005 in the vLLM batch and 0.03490 individually, exceeding the unchanged 0.02 tolerance. Real-photo accuracy and image probability calibration are unverified.
- GB10 warm single-request median latency: 59 ms at 256×256, 82 ms at 512×512, 136 ms at 768×768. These are synthetic local measurements, excluding file decoding, text encoding, HTTP and model loading; they do not establish recognition accuracy at the larger sizes.

## Local setup and experiments

Linux aarch64 / NVIDIA GB10, Python 3.12, BF16, vLLM 0.30.0. Install `uv` and a compatible NVIDIA driver first. Windows is not validated for this project.

```bash
cd /path/to/kev-vision
bash scripts/setup_kev_vision.sh
# Optional regeneration of the original 27-question text baseline:
.venv/bin/python scripts/validate_kev_vllm.py
# Reference image and control scoring; setup already creates the vision export:
OMP_NUM_THREADS=4 .venv/bin/python scripts/experiment_vision.py
# Expected exit code 1 until the known probability mismatch is resolved:
HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 OMP_NUM_THREADS=4 \
  .venv/bin/python scripts/test_kev_vision_vllm.py --vision-attention FLASH_ATTN
.venv/bin/python -m unittest discover -s tests -v
# Optional local speed measurement (model loading takes about a minute):
HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 OMP_NUM_THREADS=4 \
  .venv/bin/python scripts/benchmark_vision.py
```

The `kev_vision` package registers the experimental vision architecture plus the text reference architecture. Baseline helper scripts retain their `kev_vllm` filenames for continuity. Inputs, exports, generated images and logs live under ignored `artifacts/`; `.venv` is local and independent. No image HTTP service or UI is implemented yet.

## Next work

Localize the numerical mismatch, evaluate real images and varied resolutions, then design a validated image request API. Additional training and calibration should be driven by those results.

[Upstream attribution](THIRD_PARTY.md). No license for this project's original code has been selected yet.

The historical `docs/vision-provenance.json` records paths before extraction: `kev_vllm/` now maps to `kev_vision/`. All 26 recorded source/result/image/weight hashes were verified during migration. The original experiment commit is preserved in this repository history.
