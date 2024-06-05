import pytest
from Ex2 import computeCostAndGradient
from Ex2 import load_data

@pytest.mark.parametrize("theta, expected_cost, expected_gradients", [
    ([-10.0, 0.8, 0.08], 7.91, [0.3999, 21.1734, 22.2524])
])

def test_computeCostAndGradient(theta, expected_cost, expected_gradients):
    X, y = load_data("ex2data1.txt")
    cost, gradients = computeCostAndGradient(X, y, theta)
    assert pytest.approx(cost, rel=1e-2) == expected_cost, f"Expected cost: {expected_cost}, but got: {cost}"
    assert pytest.approx(gradients, rel=1e-2) == expected_gradients, f"Expected gradients: {expected_gradients}, but got: {gradients}"

if __name__ == "__main__":
    pytest.main()

