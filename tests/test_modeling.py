"""Unit tests for the focal-loss objective and the time-aware split."""
import numpy as np
import pandas as pd

from src import config
from src.modeling import (
    make_focal_loss_objective, focal_loss_eval, time_split, sigmoid,
)


class _FakeDataset:
    """Minimal stand-in for a lightgbm.Dataset exposing get_label()."""
    def __init__(self, label):
        self._label = np.asarray(label, dtype=float)

    def get_label(self):
        return self._label


def test_focal_grad_hess_shapes_and_hessian_positive():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 256).astype(float)
    z = rng.normal(0, 1, 256)
    obj = make_focal_loss_objective(alpha=0.25, gamma=2.0)
    grad, hess = obj(z, _FakeDataset(y))
    assert grad.shape == z.shape == hess.shape
    assert np.isfinite(grad).all() and np.isfinite(hess).all()
    # Hessian should be (near) non-negative for a sane convex-ish objective region.
    assert (hess >= -1e-6).mean() > 0.95


def test_focal_grad_sign_pushes_toward_target():
    # For a positive example with a very negative score, the gradient wrt the
    # score should be negative (boosting will move the score up).
    obj = make_focal_loss_objective(0.5, 2.0)
    grad, _ = obj(np.array([-4.0]), _FakeDataset([1.0]))
    assert grad[0] < 0


def test_focal_eval_lower_is_better_flag():
    name, value, is_higher_better = focal_loss_eval()(
        np.zeros(10), _FakeDataset(np.zeros(10)))
    assert name == "focal_loss"
    assert is_higher_better is False
    assert value >= 0


def test_time_split_is_forward_in_time():
    df = pd.DataFrame({
        config.WEEK_NUM: np.repeat(np.arange(10), 50),
        config.TARGET: np.random.default_rng(0).integers(0, 2, 500),
    })
    tr, va, te = time_split(df, valid_frac=0.2, test_frac=0.2)
    assert tr[config.WEEK_NUM].max() < va[config.WEEK_NUM].min()
    assert va[config.WEEK_NUM].max() < te[config.WEEK_NUM].min()


def test_sigmoid_bounds():
    x = np.array([-100.0, 0.0, 100.0])
    s = sigmoid(x)
    assert np.allclose(s, [0.0, 0.5, 1.0], atol=1e-6)
