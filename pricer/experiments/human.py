"""CSV input/output for the optional human price-estimation baseline."""

import csv
from pathlib import Path


def write_human_template(items, path, size=100):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows((item.summary, 0) for item in items[:size])


def read_human_pricer(items, path, size=100):
    expected = items[:size]
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    if len(rows) != len(expected):
        raise ValueError("Human predictions must match the evaluated item count")
    if any(row[0] != item.summary for row, item in zip(rows, expected)):
        raise ValueError("Human predictions must use the same products and ordering")
    predictions = {id(item): float(row[1]) for item, row in zip(expected, rows)}
    def human_pricer(item):
        return predictions[id(item)]
    return human_pricer
