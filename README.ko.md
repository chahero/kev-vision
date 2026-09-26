# kev-vision

이미지·질문·선택지를 받아 선택 결과와 점수를 계산하는 실험 프로젝트입니다. `kev-vllm`의 로컬 비전 실험을 분리했으며, 원본 Qwen3.5 비전 인코더와 Kev 언어 가중치·판단 헤드를 사용합니다. 추가 학습은 하지 않았습니다. 공식 Kev 배포물이 아닙니다.

- 합성 이미지 14개: Transformers와 vLLM 모두 14/14 정답. 빈 이미지·이미지 제거 조건은 각각 5/14.
- 기존 텍스트 27개: 원본과 선택 결과 일치.
- **확률 일치 검증은 미통과**: CAT 이미지에서 배치 약 3.0%p, 단독 약 3.49%p 차이로 기존 2%p 기준을 초과합니다.
- GB10 반복 속도 측정 중앙값: 256×256 59ms, 512×512 82ms, 768×768 136ms. 파일 읽기·텍스트 인코딩·HTTP·모델 로딩은 제외했습니다. 큰 해상도의 인식 정확도는 검증하지 않았습니다.

## 현재 장비에서 실행

```bash
cd /home/chahero/git-repository/kev-vision
.venv/bin/python -m unittest discover -s tests -v
HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 OMP_NUM_THREADS=4 \
  .venv/bin/python scripts/test_kev_vision_vllm.py --vision-attention FLASH_ATTN
```

마지막 명령은 알려진 확률 차이로 종료 코드 1을 반환하는 것이 현재 결과입니다. 실패를 통과로 처리하거나 허용 기준을 완화하지 않았습니다.

새 환경은 `bash scripts/setup_kev_vision.sh`로 준비합니다. Linux aarch64·GB10·BF16만 검증했으며 이미지 HTTP API·UI·Windows 지원은 아직 없습니다. 현재 모델과 자료는 이전 프로젝트에서 복사·이전했고, 가상환경은 이 폴더에 새로 설치했습니다.

[상세 실험 기록](docs/vision-experiment.md) · [설치와 범위](README.md) · [외부 구성요소](THIRD_PARTY.md)
