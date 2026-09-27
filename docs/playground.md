# Image Playground / 이미지 체험

Windows: `run_vision.bat`. Linux: `.venv/bin/python scripts/vision_playground.py --open`.
Prepare the model first. The browser opens after model loading completes.
UI: <http://127.0.0.1:18092>; interactive API docs: <http://127.0.0.1:18092/docs>.
Run one instance; one request is processed at a time. Stop with Ctrl+C.

1. 예제 또는 업로드 파일 선택 / Select an example or upload an image.
2. 질문·선택지 수정 / Edit the question and options.
3. 실행 후 확률·JSON 확인 / Run and inspect the actual probabilities and JSON.

The Korean/English toggle translates UI labels, not input text. Examples use English.
PNG, JPEG and WEBP are accepted, up to 10 MB and 20 megapixels. Files are decoded
locally, orientation-corrected, aspect-preserving and padded to 256×256. Small text
and details can be lost. Uploads are not persisted by the application or sent to a
cloud model. There is no upload history. This loopback service is for local testing.

Use 2–12 distinct nonempty options. Image plus state: at most 1,024 tokens;
full input: at most 2,048 tokens. Editing input clears the previous result.
Probabilities are relative scores among the supplied choices, not calibrated
confidence. Include an unknown option when appropriate.

## Upload API

`POST /api/decide` accepts multipart fields `image` (file), `state`, `question`,
and `options` (a JSON array encoded as text). This is the playground API, not the
native vLLM pooling API. `GET /api/health` and `GET /api/examples` are also available.

```python
import json
import httpx

with open("examples/images/color-red.png", "rb") as image:
    response = httpx.post("http://127.0.0.1:18092/api/decide",
        files={"image": ("color-red.png", image, "image/png")},
        data={"state": "Inspect the supplied image.",
              "question": "What color is the large shape?",
              "options": json.dumps(["Red", "Green", "Blue"])}, timeout=180)
response.raise_for_status()
print(response.json())
```

Response: `selected`, zero-based `selected_index`, `probabilities`, `tokens`,
`elapsed_ms`, `input_size`, `vision_precision`. Model loading is excluded from
request timing. Errors: 413 oversized request, 422 invalid image/input, 429 busy,
503 model unavailable, 500 inference failure (inspect the launch console).

Run all five examples while the UI is running:

```powershell
.\.venv\Scripts\python.exe examples/run_examples.py
```

The application uses the existing Python environment and static HTML/CSS/JS;
no Node build is required. The Windows upload flow was exercised in the in-app
browser; Linux UI execution has not been revalidated in this update.
