"""Offline held-out pricing comparison on a versioned JSONL product snapshot."""
import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge


def evaluate(path):
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    ids, texts = set(), set()
    for row in rows:
        text = ' '.join(row['text'].lower().split())
        if row['id'] in ids or text in texts:
            raise ValueError('Duplicate product ID or normalized text')
        if not text or not math.isfinite(row['price']) or row['price'] <= 0:
            raise ValueError('Nonempty text and positive finite prices required')
        ids.add(row['id'])
        texts.add(text)
    # Product IDs determine membership, independently of row order and labels.
    train = [r for r in rows if int(hashlib.sha256(r['id'].encode()).hexdigest(), 16) % 5 != 0]
    test = [r for r in rows if int(hashlib.sha256(r['id'].encode()).hexdigest(), 16) % 5 == 0]
    if not train or not test:
        raise ValueError('Both splits must be nonempty')
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, sublinear_tf=True)
    started = time.perf_counter()
    features = vectorizer.fit_transform([r['text'] for r in train])
    model = Ridge(alpha=10, solver='lsqr').fit(features, [r['price'] for r in train])
    fit_seconds = time.perf_counter() - started
    mean = float(np.mean([r['price'] for r in train]))
    results = {}
    predictions = []
    for name in ('training_mean', 'tfidf_ridge'):
        errors, elapsed = [], []
        failed = 0
        for row in test:
            start = time.perf_counter()
            value = mean if name == 'training_mean' else max(0., float(model.predict(vectorizer.transform([row['text']]))[0]))
            elapsed.append((time.perf_counter() - start) * 1000)
            if not math.isfinite(value):
                failed += 1
                continue
            errors.append(value - row['price'])
            predictions.append({'configuration': name, 'id': row['id'], 'actual': row['price'], 'predicted': value})
        results[name] = {'attempted': len(test), 'valid': len(errors), 'failure_rate': failed / len(test),
                         'mae': float(np.mean(np.abs(errors))), 'rmse': float(np.sqrt(np.mean(np.square(errors)))),
                         'wape': float(np.sum(np.abs(errors)) / sum(r['price'] for r in test)) if not failed else None,
                         'latency_ms_mean': float(np.mean(elapsed)), 'latency_ms_p95': float(np.percentile(elapsed, 95))}
    return {'schema_version': 1, 'dataset_sha256': hashlib.sha256(raw).hexdigest(),
            'split': 'sha256(product_id) modulo 5: 0=test; otherwise=train',
            'dataset_size': len(rows), 'train_size': len(train), 'test_size': len(test),
            'configuration': {'ngram_range': [1, 2], 'max_features': 20000, 'alpha': 10, 'solver': 'lsqr'},
            'environment': {'python': platform.python_version(), 'sklearn': sklearn.__version__, 'platform': platform.platform()},
            'fit_seconds': fit_seconds, 'results': results, 'predictions': predictions,
            'limitations': 'Historical catalog prices, one category, exact ID/text deduplication only; related variants may cross splits. This measures text regression, not the live ensemble or retrieval relevance.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('benchmarks/evaluation/appliances.jsonl'))
    parser.add_argument('--output', type=Path, default=Path('benchmarks/results/pricing-evidence.json'))
    args = parser.parse_args()
    report = evaluate(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'predictions'}, indent=2))


if __name__ == '__main__':
    main()
