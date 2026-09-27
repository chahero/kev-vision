# 비전 정밀도 검증 결과

2026-09-27, Linux aarch64 / NVIDIA GB10 / BF16 / eager 모드.
**`--vision-precision reference` 후보는 현재 합성 이미지·텍스트 회귀에서 기존
2%p 확률 허용치를 통과했습니다.** 기본 `native` 경로는 계속 미통과이며,
새 후보는 명시적인 옵션으로 활성화합니다.

## 구현

[`kev_vision/precision.py`](../kev_vision/precision.py)는 현재 이미지 격자와
체크포인트 가중치로 다음 계산을 수행합니다.

1. 패치 변환을 행렬곱 단축 경로 대신 Conv3d로 실행합니다.
2. 위치 임베딩 보간의 가중치·곱·합을 FP32로 유지하고 최종 합만 BF16으로 변환합니다.
3. FP32 RoPE 주파수·cos/sin을 새로 계산하고, 비전 블록 24개 모두의 회전 계산을
   FP32로 수행한 뒤 Q/K를 BF16으로 돌려줍니다.

기준 텐서·진단 캡처·정답·파일명은 읽지 않습니다. 가중치 추가 학습이나 변경은
없으며 attention과 언어 모델은 기존 vLLM 경로입니다. RoPE 도우미에는 학습
파라미터나 영구 체크포인트 버퍼가 없습니다. 설정은 런타임
`hf_overrides={"kev_vision_precision": "reference"}`로 전달하며 모델 파일은
수정하지 않습니다. 현재 옵션은 eager 실행을 요구합니다.

## 검증 결과

합성 이미지 14개, 빈 이미지 대조군 14개, 이미지 없는 대조군 14개, 기존 텍스트
27개로 총 69개 입력입니다. 정순 배치·역순 배치·개별 단독 실행을 비교했습니다.
후보는 이미지 처리 캐시를 끈 경우와 기본 4 GiB 캐시를 사용하는 경우 모두
**207/207**, 합계 **414/414 비교**에서 기준과 선택 결과가 같고 확률 오차가
0.02 미만이었습니다. 각 실행의 초기 CAT 단독 확인도 별도로 통과했습니다.

아래 수치는 모든 선택지 중 최대 절대 확률 오차입니다.

| 실행 | 기본 경로, 캐시 끔 | 후보, 캐시 끔 | 후보, 기본 캐시 |
| --- | ---: | ---: | ---: |
| 정순 배치 | 4.3130%p | 1.4982%p | 0.8309%p |
| 역순 배치 | 4.3864%p | 1.2158%p | 0.8357%p |
| 단독 | 3.4902%p | 1.0237%p | 1.0237%p |
| 최종 판정 | 실패 | **통과** | **통과** |

- 이미지 정답은 각 조건에서 14/14입니다. 빈 이미지와 이미지 없는 대조군은
  각각 5/14이며, 정답을 판단하기 위해 이미지가 사용되는 기존 관찰을 유지합니다.
- 텍스트 27개는 모든 모드에서 선택 결과가 일치하고 최대 기준 오차는 약 0.8064%p입니다.
- CAT 단독 확률은 기준 69.335955%, 기본 경로 72.826111%, 후보 70.264006%입니다.
  후보의 CAT 단독 오차는 0.928050%p입니다.
- 후보의 최악 사례는 캐시를 끈 정순 배치의 `position-left`로 1.498163%p입니다.
- 허용치를 완화하지 않았습니다. 입력 토큰 일치, 출력 형태, 유한성, 확률 범위와
  합계, 선택 결과, 확률 오차를 검사합니다.

언어 가중치와 실행 구현을 바꾸지는 않았지만 혼합 배치의 수치는 반드시 비트
단위로 같지는 않습니다. 기본 경로와 후보 사이 텍스트 확률의 최대 차이는
정순 혼합 배치에서 0.4058%p, 역순과 단독에서는 0이었습니다. 캐시 설정에 따른
배치 수치도 달라집니다. 이 관찰은 각 모드를 독립적으로 기준과 대조해야 하는
이유이며, 특정 스케줄러나 커널 원인까지 확정한 것은 아닙니다.

기본 경로와 정밀도 정렬 경로는 초기 CAT 요청, 캐시 끔, 역순·단독 실행을
포함한 같은 조건으로 비교했습니다.

## 속도

계측 훅 없이 같은 256×256 CAT 입력과 문항으로 측정했습니다. 이미지 한쪽
구석 픽셀을 매번 바꿔 캐시 재사용을 방지하고 이미지 처리 캐시는 껐습니다.
준비된 PIL 이미지·토큰에서 `LLM.encode`를 거쳐 CPU 확률을 받을 때까지이며,
파일 읽기·입력 준비·모델 로딩·HTTP·워밍업은 제외합니다.

| 요청 묶음 크기 | 기본 경로 중앙값 | 후보 중앙값 | 비율 |
| --- | ---: | ---: | ---: |
| 1 | 60.575ms | 61.480ms | 1.015배 |
| 2 | 72.847ms | 75.895ms | 1.042배 |

각 크기별 워밍업 5회 후 단독 30회, 2개 묶음 20회를 측정했습니다.
다른 서비스가 상주한 장비의 로컬 1회 측정으로 일반적인 성능 보장은 아닙니다.

## 재현

현재 준비된 모델·이미지·기준값이 있는 저장소 루트에서 실행합니다.

```bash
# 기본 캐시 사용: 후보 전체 회귀. 현재 결과는 종료 코드 0.
HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 OMP_NUM_THREADS=4 \
  .venv/bin/python scripts/test_kev_vision_vllm.py \
  --vision-attention FLASH_ATTN --vision-precision reference --all-singletons \
  --output artifacts/vision-experiment/reference-precision-default-cache-result.json

# 후보 캐시 끔 + 속도 측정. 현재 결과는 종료 코드 0.
HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 OMP_NUM_THREADS=4 \
  .venv/bin/python scripts/test_kev_vision_vllm.py \
  --vision-attention FLASH_ATTN --vision-precision reference --all-singletons --benchmark

# 동일 조건의 기본 경로. CAT 확률 오차로 종료 코드 1이 예상됨.
HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 OMP_NUM_THREADS=4 \
  .venv/bin/python scripts/test_kev_vision_vllm.py \
  --vision-attention FLASH_ATTN --vision-precision native --all-singletons --benchmark

.venv/bin/python -m unittest discover -s tests -v
```

CPU 테스트는 기존 입력 검사 3개와 새 설정 호환성·가중치 보존·혼합 격자 경계
검사 2개로 총 5개가 통과했습니다. 새 GPU 프로세스에서 모델 로딩·워밍업부터
정상 실행했으며 진단 RPC나 기준 중간값 주입은 사용하지 않았습니다.

## 범위와 자료

이번 통과는 단일 이미지, 256×256 합성 입력, BF16, 최대 2048 토큰, eager,
동시 시퀀스 2개 설정에 한정됩니다. 실제 사진·문서, 큰 이미지의 정확도,
이미지 확률 보정, Windows, 컴파일·CUDA graph, HTTP/UI는 검증하지 않았습니다.
원본 체크포인트 내보내기는 유지하고 기본 실행도 `native`로 유지합니다.

[비교 요약·버전·소스 해시](vision-precision-comparison.json) ·
[후보 기본 캐시 결과](vision-reference-precision-result.json) ·
[후보 캐시 끔·속도 결과](vision-reference-precision-benchmark-result.json) ·
[기본 경로 비교 결과](vision-native-precision-extended-result.json)
