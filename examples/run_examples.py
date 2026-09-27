"""Run all five image examples against a running local playground."""
import argparse
import json
from pathlib import Path
import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:18092')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    cases = json.loads((root / 'decisions.json').read_text(encoding='utf-8'))
    results = []
    with httpx.Client(timeout=180) as client:
        for case in cases:
            path = root / 'images' / Path(case['image']).name
            with path.open('rb') as file:
                response = client.post(args.url.rstrip('/') + '/api/decide',
                    data={'state': case['state'], 'question': case['question'], 'options': json.dumps(case['options'])},
                    files={'image': (path.name, file, 'image/png')})
            response.raise_for_status()
            result = response.json()
            passed = result['selected_index'] == case['expected_index']
            results.append({'id': case['id'], 'passed': passed, 'response': result})
            print(f"{'PASS' if passed else 'FAIL'} {case['id']}: {result['selected']}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2)+'\n', encoding='utf-8')
    raise SystemExit(0 if all(r['passed'] for r in results) else 1)


if __name__ == '__main__':
    main()
