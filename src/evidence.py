"""Compare inference responses using the same tolerance as workflow.check."""
import math

RTOL = 1e-10
ATOL = 1e-9


def prediction_responses_match(left, right):
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    if {k: v for k, v in left.items() if k != 'predictions_MW'} != {k: v for k, v in right.items() if k != 'predictions_MW'}:
        return False
    a, b = left.get('predictions_MW'), right.get('predictions_MW')
    if not isinstance(a, list) or not isinstance(b, list) or not a or len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if isinstance(x, bool) or isinstance(y, bool) or not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            return False
        if not math.isfinite(x) or not math.isfinite(y) or abs(x - y) > ATOL + RTOL * abs(y):
            return False
    return True
