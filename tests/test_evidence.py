import pytest
from evidence import prediction_responses_match


def response(values=None, **extra):
    return {'model_version': '2', 'unit': 'MW', 'predictions_MW': [465.65352000000087] if values is None else values, **extra}


def test_parallel_rf_rounding_is_not_a_failed_update():
    assert prediction_responses_match(response(), response([465.6535200000009]))


@pytest.mark.parametrize('other', [
    response([466.65352]), response(model_version='1'), response(unit='kW'),
    response([]), response([465.65352, 470.0]), response([float('nan')]), response(['465.65352']),
])
def test_real_response_mismatches_are_rejected(other):
    assert not prediction_responses_match(response(), other)
