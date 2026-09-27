# Native Windows / Windows 실행

This project uses its own `.venv`; do not copy a virtual environment from another
project or machine. Tested: Python 3.13.2, NVIDIA TITAN RTX 24 GiB (SM 7.5),
PyTorch 2.13.0+cu130, Transformers 5.17.0, PEFT 0.21.0 and the pinned community
vLLM 0.29.0 Windows wheel. The Linux profile remains vLLM 0.30.0/BF16.

## Setup and run

```powershell
.\scripts\setup_kev_vision.ps1
.\run_vision.bat
```

Install `uv` and a compatible NVIDIA driver first. Allow several tens of GB.
Setup installs packages, downloads pinned inputs, prepares the text merge and
creates the FP16 vision export. The first run can take several minutes.
`-Offline` applies to model fetching; package installation can still require network access.

The tested wheel and SHA256 are listed in `requirements-windows.txt`.
This is a community native Windows build, not an official PyPI Windows wheel.
Setup can reuse completed exports. If export was interrupted, preserve or move
the incomplete output before rebuilding; the exporter refuses to overwrite weights.

| Local path | Contents |
| --- | --- |
| `.venv` | Dedicated Python environment |
| `artifacts/hf-cache` | Upstream model cache |
| `artifacts/windows-vllm-audit` | Pinned Kev source/checkpoint, FP16 text export and references |
| `artifacts/windows-vision-experiment` | FP16 vision export, generated fixtures and results |

During this PC's setup, existing pinned inputs and the text export were copied
into this repository, while the immutable base snapshot cache was reused from
kev-vllm. Neither its environment nor its installed plugin is used. A fresh setup
downloads its own inputs and requires no sibling repository. To rebuild the base
without the reused cache, rerun the fetch script with this project's `HF_HOME`.

The UI uses port 18092 and loads the model once. Keep the console open and use
Ctrl+C to stop. Do not start a second GPU test while the UI is running.
`KEV_GPU_MEMORY_UTILIZATION` defaults to 0.75 on Windows; ensure enough free VRAM.
The TITAN RTX has no native BF16 support, so this profile exports and runs FP16
and uses TORCH_SDPA for vision attention. Text attention uses TRITON_ATTN.

CLI (loads a new model per invocation):

```powershell
.\.venv\Scripts\python.exe scripts/query_kev_vision.py --image examples/images/color-red.png --question "What color is the large shape?" --options Red Green Blue --output artifacts/my-result.json
```

## Reproduce validation

Stop the UI first. Run sequentially from the project root:

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:VLLM_NO_USAGE_STATS = '1'
$env:OMP_NUM_THREADS = '4'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/validate_kev_vllm.py --phase reference
.\.venv\Scripts\python.exe scripts/experiment_vision.py
.\.venv\Scripts\python.exe scripts/test_kev_vision_vllm.py --vision-attention TORCH_SDPA --vision-precision reference --all-singletons --output artifacts/windows-vision-experiment/windows-result.json
```

2026-09-27 results: 12 images + 12 blank + 12 no-image controls + 27 text rows =
63 inputs. Forward/reversed/singleton comparisons: 189/189 passed with identical
argmax. Maximum absolute probability error: 0.00103465 (0.103465 percentage points),
below the unchanged 0.02 tolerance. Actual images: 12/12 correct; controls: 4/12 each.
The initial singleton diagnostic also passed. This run used the default image cache;
cache-disabled Windows benchmarking was not performed.

All 13 unit/API tests passed without skips. The five bundled images also passed
through the persistent UI's real multipart upload API. See
[numerical summary](windows-validation.json) and [API responses](playground-results.json).
The Linux DejaVu font was unavailable, so CAT/DOG fixtures were omitted rather
than silently replacing the font and claiming the same 14-case suite.
Real-photo/document accuracy and calibration remain unverified.

## Browser and visual QA

The in-app browser exercised example selection, real inference, local file upload
and preview replacement, Korean/English switching, JSON expansion/copy confirmation,
option add/remove, duplicate validation and clearing stale results after edits.
The 1505x1045 desktop target and 390x844 mobile target had no page horizontal
 overflow; the example rail intentionally scrolls on narrow screens. No app
console warnings/errors were observed. The browser clipboard bridge did not
return copied text, so clipboard content was not independently verified beyond
the browser success state.

Concept and implementation screenshots were inspected with view_image. Comparison
points: centered title/copy, three-column hierarchy, cool gray/white/blue palette,
input and result type hierarchy, actual image framing, and mobile stacking.
The design was faithfully implemented with these intentional differences: existing
playground letter labels/icons and header are reused for consistency; fixture
thumbnails use exact test inputs; a fifth real example replaces the conceptual
upload shortcut; upload uses the native file chooser; JSON starts collapsed;
probabilities/timings are measured values. No material visual mismatch remains.
