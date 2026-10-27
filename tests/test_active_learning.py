import numpy as np
import pandas as pd
import pytest

from immune_response.active_learning import ridge_leverage, run_active_learning


def test_leverage_uses_design_and_falls_with_additional_measurements():
    labeled = np.eye(2)
    pool = np.eye(2)
    before = ridge_leverage(labeled, pool, alpha=1)
    after = ridge_leverage(np.vstack([labeled, [1, 0]]), pool, alpha=1)
    assert after[0] < before[0]
    assert after[1] == before[1]


def test_acquisition_shares_initial_conditions_and_separates_evaluation():
    meta = pd.DataFrame([(drug, cell) for drug in "abcde" for cell in "TBNK"],
                        columns=["drug", "cell_type"])
    y = np.random.default_rng(4).normal(size=(len(meta), 7))
    result = run_active_learning(meta, y, seeds=[1, 2], n_rounds=2, batch_size=2)
    changed_labels = run_active_learning(meta, y[::-1], seeds=[1, 2], n_rounds=2, batch_size=2)
    assert set(result.attrs["evaluation_indices"]).isdisjoint(result.attrs["development_indices"])
    assert result.attrs["acquisition_history"] == changed_labels.attrs["acquisition_history"]
    assert np.isfinite(result.rmse).all()
    for seed in (1, 2):
        random = result.query("seed == @seed and strategy == 'random'").iloc[0]
        leverage = result.query("seed == @seed and strategy == 'ridge_leverage'").iloc[0]
        assert random.n_labeled == leverage.n_labeled
        assert random.rmse == leverage.rmse


def test_explicit_evaluation_indices_and_invalid_splits():
    meta = pd.DataFrame([(drug, cell) for drug in "abcde" for cell in "TBNK"],
                        columns=["drug", "cell_type"])
    y = np.random.default_rng(4).normal(size=(len(meta), 7))
    result = run_active_learning(meta, y, seeds=[1], n_rounds=1, batch_size=2,
                                 evaluation_indices=[3, 7, 11])
    assert result.attrs["evaluation_indices"] == [3, 7, 11]
    for history in result.attrs["acquisition_history"].values():
        selected = history["initial"] + sum(history["batches"], [])
        assert not set(selected) & {3, 7, 11}
    for invalid in ([-1], [1, 1], [1.1], [20], [3, 7, 11, 15, 19]):
        with pytest.raises(ValueError):
            run_active_learning(meta, y, evaluation_indices=invalid)
