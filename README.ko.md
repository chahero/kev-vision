# kev-vision

이미지·질문·선택지를 받아 선택 결과와 점수를 계산하는 실험 프로젝트입니다. Qwen3.5 비전 인코더, Kev 언어 가중치·판단 헤드, vLLM을 사용합니다. 추가 학습은 하지 않았으며 공식 Kev 배포물이 아닙니다.

[English](README.md) · [검증 결과](docs/validation.md) · [외부 구성요소](THIRD_PARTY.md)

## 실행

Linux aarch64 / NVIDIA GB10, Python 3.12, BF16, eager 환경에서 검증했습니다. `uv`와 호환 NVIDIA 드라이버를 준비한 뒤 저장소 루트에서 실행합니다.

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

검증된 계산 경로는 `--vision-precision reference`로 선택합니다. 직접 연동할 때는 vLLM에 `hf_overrides={"kev_vision_precision": "reference"}`와 eager 설정을 전달합니다. 옵션을 생략한 기본 `native` 경로는 확률 오차 기준을 통과하지 못합니다.

## 검증 범위

- 69개 입력 × 정순·역순·단독 × 캐시 켬·끔: 총 414개 비교 통과. 선택 결과 일치, 최대 확률 오차 1.50%p로 2%p 기준 충족.
- 합성 이미지 14/14 정답, 빈 이미지·이미지 제거 대조군 각각 5/14. 텍스트 27개 선택 결과 일치.
- 256px 단독 요청 중앙값 61.5ms, 기본 경로 60.6ms. 로딩·입력 준비·HTTP를 제외한 로컬 측정.

실제 사진·문서, 다양한 해상도의 정확도와 이미지 확률 보정은 아직 검증하지 않았습니다. 이미지 HTTP API와 UI는 없습니다.

모델·입력·생성 이미지는 로컬 `artifacts/`, 가상환경은 `.venv/`에 두며 업로드에서 제외합니다. 설치 스크립트가 고정된 버전의 원본 자료를 준비합니다. 자체 코드 라이선스는 아직 선택하지 않았으며 외부 구성요소에는 각 원본 조건이 적용됩니다.

별도 `kev-vllm` 저장소나 패키지는 필요하지 않습니다. 일부 스크립트 파일명에만 `kev_vllm`이 남아 있으며 구현은 이 프로젝트 안에 있습니다. 설치 시 고정 버전의 원본 Kev 인코더·체크포인트 소스와 가중치를 다운로드하고, 텍스트 가중치 병합 결과를 비전 모델 생성에 사용합니다.
