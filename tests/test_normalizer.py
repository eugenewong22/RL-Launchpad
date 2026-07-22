import numpy as np
import torch

from src.agent.normalizer import RunningNormalizer


def test_stats_converge_to_data_distribution():
    rng = np.random.default_rng(0)
    norm = RunningNormalizer(dim=4)
    data = rng.normal(loc=[10.0, -5.0, 0.0, 100.0], scale=[2.0, 0.5, 1.0, 30.0], size=(5000, 4))
    for chunk in np.split(data, 50):
        norm.update(chunk.astype(np.float32))
    out = norm.normalize(data.astype(np.float32))
    assert np.abs(out.mean(axis=0)).max() < 0.1
    assert np.abs(out.std(axis=0) - 1.0).max() < 0.1


def test_normalize_clips_outliers():
    norm = RunningNormalizer(dim=2)
    norm.update(np.random.default_rng(0).normal(size=(1000, 2)).astype(np.float32))
    out = norm.normalize(np.array([[1e6, -1e6]], dtype=np.float32))
    assert np.all(np.abs(out) <= 5.0)


def test_normalize_handles_torch_tensors():
    norm = RunningNormalizer(dim=3)
    norm.update(np.random.default_rng(0).normal(loc=7.0, size=(1000, 3)).astype(np.float32))
    t = torch.full((8, 3), 7.0)
    out = norm.normalize(t)
    assert isinstance(out, torch.Tensor)
    assert out.abs().mean() < 0.5  # near the running mean -> near zero


def test_save_load_roundtrip(tmp_path):
    norm = RunningNormalizer(dim=4)
    norm.update(np.random.default_rng(0).normal(loc=3.0, size=(500, 4)).astype(np.float32))
    path = tmp_path / "normalizer.npz"
    norm.save(path)
    loaded = RunningNormalizer.load(path)
    x = np.random.default_rng(1).normal(size=(10, 4)).astype(np.float32)
    assert np.allclose(norm.normalize(x), loaded.normalize(x))


def test_identity_before_any_update():
    norm = RunningNormalizer(dim=3)
    x = np.array([[1.0, -2.0, 3.0]], dtype=np.float32)
    assert np.allclose(norm.normalize(x), x)  # no stats yet -> pass-through
