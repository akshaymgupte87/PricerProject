"""Legacy notebook adapter for :mod:`pricer.evaluator`.

The original notebooks import ``evaluate`` from this root module and expect
sequential execution. All evaluation logic now lives in ``pricer.evaluator``;
this adapter preserves that public interface with one worker.
"""

from pricer.evaluator import DEFAULT_SIZE, Tester as _CanonicalTester


class Tester(_CanonicalTester):
    """Backward-compatible evaluator that defaults to sequential execution."""

    def __init__(self, predictor, data, title=None, size=DEFAULT_SIZE):
        super().__init__(predictor, data, title=title, size=size, workers=1)


def evaluate(function, data, size=DEFAULT_SIZE):
    Tester(function, data, size=size).run()


__all__ = ["DEFAULT_SIZE", "Tester", "evaluate"]
