import pytest
from aeons.aeons import logXf_formula


def test_logXf_formula():
    theta = [0, 10, 1]
    logZdead, Xi, epsilon = 1, 1, 1e-3
    assert logXf_formula(theta, logZdead, Xi, epsilon) == pytest.approx(-5.553156483219841)
