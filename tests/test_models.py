import numpy as np
import pandas as pd
import pytest

from immune_response.models import AdditiveRidge, DrugMeanBaseline, PCARidge, TorchInteractionModel, ZeroBaseline


@pytest.fixture
def profiles():
    meta = pd.DataFrame({"drug": ["a", "a", "b", "b", "c", "c"],
                         "cell_type": ["T", "B"] * 3})
    rng = np.random.default_rng(3)
    return meta, rng.normal(size=(6, 8)).astype(np.float32)


def test_drug_mean_uses_only_available_training_profiles(profiles):
    meta, y = profiles
    model = DrugMeanBaseline().fit(meta.iloc[[0, 2, 3]], y[[0, 2, 3]])
    predicted = model.predict(meta.iloc[[1, 4]])
    np.testing.assert_allclose(predicted[0], y[0])
    np.testing.assert_allclose(predicted[1], y[[0, 2, 3]].mean(axis=0))


@pytest.mark.parametrize("model", [ZeroBaseline(), DrugMeanBaseline(), AdditiveRidge(), PCARidge(n_components=3)])
def test_models_return_finite_gene_profiles(model, profiles):
    meta, y = profiles
    predicted = model.fit(meta, y).predict(meta.iloc[:2])
    assert predicted.shape == (2, 8)
    assert np.isfinite(predicted).all()


def test_invalid_responses_rejected(profiles):
    meta, y = profiles
    y[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        AdditiveRidge().fit(meta, y)


def test_pca_ridge_uses_only_training_targets(profiles):
    meta, y = profiles
    model = PCARidge(n_components=3).fit(meta.iloc[:4], y[:4])
    np.testing.assert_allclose(model.pca_.mean_, y[:4].mean(axis=0), atol=1e-6)
    assert model.pca_.n_components_ == 3
    assert model.predict(meta.iloc[4:]).shape == (2, 8)


def test_torch_training_pca_and_reproducibility(profiles):
    pytest.importorskip("torch")
    meta, y = profiles
    kwargs = dict(seed=7, epochs=8, hidden=8, n_components=3)
    first = TorchInteractionModel(**kwargs).fit(meta.iloc[:4], y[:4])
    second = TorchInteractionModel(**kwargs).fit(meta.iloc[:4], y[:4])
    np.testing.assert_allclose(first.pca_.mean_, y[:4].mean(axis=0), atol=1e-6)
    np.testing.assert_allclose(first.predict(meta.iloc[4:]), second.predict(meta.iloc[4:]))
    assert first.predict(meta.iloc[4:]).shape == (2, 8)
