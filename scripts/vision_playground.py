"""Local image-upload playground. Model is loaded once before opening the browser."""
import argparse
import io
import json
import threading
import warnings
import webbrowser
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from PIL import Image, UnidentifiedImageError
from kev_runtime import ROOT

MAX_IMAGE_BYTES = 10 * 1024 * 1024


def read_image(data):
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError('Image must be between 1 byte and 10 MB')
    with warnings.catch_warnings():
        warnings.simplefilter('error', Image.DecompressionBombWarning)
        try:
            source = Image.open(io.BytesIO(data))
            if source.format not in ('PNG', 'JPEG', 'WEBP'):
                raise ValueError('Use PNG, JPEG or WEBP')
            if source.width * source.height > 20_000_000:
                raise ValueError('Image must contain at most 20 megapixels')
            source.load()
            return source
        except (UnidentifiedImageError, OSError, Image.DecompressionBombWarning, Image.DecompressionBombError) as error:
            raise ValueError('Cannot decode this image') from error


def create_app(engine=None):
    app = FastAPI(title='Kev Vision Playground')
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost', 'testserver'])
    lock = threading.Lock()

    @app.middleware('http')
    async def local_only(request: Request, call_next):
        if request.method == 'POST':
            origin = request.headers.get('origin')
            if origin and origin != str(request.base_url).rstrip('/'):
                return JSONResponse({'detail': 'Cross-origin requests are not accepted'}, status_code=403)
            size = 0
            chunks = []
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_IMAGE_BYTES + 65536:
                    return JSONResponse({'detail': 'Request exceeds 10 MB image limit'}, status_code=413)
                chunks.append(chunk)
            request._body = b''.join(chunks)
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Cache-Control'] = 'no-store'
        return response

    @app.get('/', include_in_schema=False)
    def home():
        return FileResponse(ROOT / 'playground/index.html')

    @app.get('/api/health')
    def health():
        return {'ready': engine is not None, 'status': 'ready' if engine else 'offline'}

    @app.get('/api/examples')
    def examples():
        return json.loads((ROOT / 'examples/decisions.json').read_text(encoding='utf-8'))

    @app.post('/api/decide')
    def decide(image: UploadFile = File(...), state: str = Form(...), question: str = Form(...), options: str = Form(...)):
        state, question = state.strip(), question.strip()
        try:
            choices = json.loads(options)
            if not isinstance(choices, list) or not 2 <= len(choices) <= 12 or not all(isinstance(o, str) for o in choices):
                raise ValueError('Supply 2–12 text options')
            choices = [o.strip() for o in choices]
            if any(not o or len(o) > 1000 for o in choices) or len(set(choices)) != len(choices):
                raise ValueError('Options must be distinct and nonempty')
            if not state or not question or len(state) > 16000 or len(question) > 4000:
                raise ValueError('Invalid state or question length')
            picture = read_image(image.file.read(MAX_IMAGE_BYTES + 1))
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        finally:
            image.file.close()
        if engine is None:
            picture.close()
            raise HTTPException(503, 'Model is not ready')
        if not lock.acquire(blocking=False):
            picture.close()
            raise HTTPException(429, 'Another request is running')
        try:
            return engine.decide(picture, state, question, choices)
        except Exception as error:
            if type(error).__name__ == 'ContextOverflow':
                raise HTTPException(422, 'Input too long: image + state 1024 tokens, full row 2048 tokens') from error
            raise HTTPException(500, 'Model inference failed; see the launch console') from error
        finally:
            picture.close()
            lock.release()

    app.mount('/assets', StaticFiles(directory=ROOT / 'playground'), name='assets')
    app.mount('/examples', StaticFiles(directory=ROOT / 'examples'), name='examples')
    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=18092)
    parser.add_argument('--open', action='store_true')
    args = parser.parse_args()
    import uvicorn
    from vision_service import VisionEngine
    print('Loading Kev Vision; first startup may take several minutes.', flush=True)
    engine = VisionEngine()
    url = f'http://127.0.0.1:{args.port}'
    print(f'Ready: {url}', flush=True)
    if args.open:
        timer = threading.Timer(1.5, lambda: webbrowser.open(url))
        timer.daemon = True
        timer.start()
    uvicorn.run(create_app(engine), host='127.0.0.1', port=args.port)
