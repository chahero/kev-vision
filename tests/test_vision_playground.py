import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from fastapi.testclient import TestClient
from PIL import Image
from vision_playground import create_app


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.engine = Mock()
        self.engine.decide.return_value = {'selected': 'Red', 'selected_index': 0,
                                         'probabilities': [{'option': 'Red', 'probability': 1.0}, {'option': 'Blue', 'probability': 0.0}]}
        self.client = TestClient(create_app(self.engine))
        buffer = io.BytesIO()
        Image.new('RGB', (20, 40), 'red').save(buffer, format='PNG')
        self.image = buffer.getvalue()

    def post(self, **overrides):
        body = dict(state='Inspect.', question='Color?', options=json.dumps(['Red', 'Blue']))
        body.update(overrides)
        return self.client.post('/api/decide', data=body, files={'image': ('input.png', self.image, 'image/png')})

    def test_actual_image_reaches_engine(self):
        response = self.post()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['selected'], 'Red')
        self.assertEqual(self.engine.decide.call_args.args[0].size, (20, 40))

    def test_duplicate_options_rejected(self):
        self.assertEqual(self.post(options='["Red", " Red "]').status_code, 422)
        self.engine.decide.assert_not_called()

    def test_corrupt_image_rejected(self):
        self.image = b'not an image'
        self.assertEqual(self.post().status_code, 422)
        self.engine.decide.assert_not_called()

    def test_unsupported_format_rejected(self):
        buffer = io.BytesIO()
        Image.new('RGB', (4, 4)).save(buffer, format='BMP')
        self.image = buffer.getvalue()
        self.assertEqual(self.post().status_code, 422)

    def test_cross_origin_rejected(self):
        response = self.client.post('/api/decide', headers={'Origin': 'https://example.com'})
        self.assertEqual(response.status_code, 403)

    def test_missing_image_rejected(self):
        self.assertEqual(self.client.post('/api/decide', data={'state': 'x'}).status_code, 422)

    def test_input_overflow_is_readable(self):
        class ContextOverflow(Exception):
            pass
        self.engine.decide.side_effect = ContextOverflow()
        response = self.post()
        self.assertEqual(response.status_code, 422)
        self.assertIn('1024', response.json()['detail'])

    def test_offline_and_examples(self):
        self.assertTrue(self.client.get('/api/health').json()['ready'])
        self.assertEqual(len(self.client.get('/api/examples').json()), 5)
        self.client = TestClient(create_app())
        self.assertFalse(self.client.get('/api/health').json()['ready'])
        self.assertEqual(self.post().status_code, 503)


if __name__ == '__main__':
    unittest.main()
