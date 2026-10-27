import unittest

import numpy as np
import pandas as pd

from immune_response.evaluation import make_splits, score_predictions, bootstrap_rmse_improvement


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.meta = pd.DataFrame(
            [(f"drug_{d}", c) for d in range(8) for c in ("B", "T", "NK")],
            columns=["drug", "cell_type"],
        )

    def test_pair_holdouts_preserve_categories_and_cover_every_row(self):
        # Non-default indices must never be mistaken for positional indices.
        meta = self.meta.sample(frac=1, random_state=1).set_index(np.arange(100, 124))
        splits = make_splits(meta, seed=12)
        repeated = make_splits(meta, seed=12)
        held_out = []
        for (train, test), (train2, test2) in zip(splits, repeated):
            np.testing.assert_array_equal(train, train2)
            np.testing.assert_array_equal(test, test2)
            self.assertFalse(set(map(tuple, meta.iloc[train].values)) & set(map(tuple, meta.iloc[test].values)))
            for column in meta:
                self.assertLessEqual(set(meta.iloc[test][column]), set(meta.iloc[train][column]))
            held_out.extend(test)
        self.assertEqual(sorted(held_out), list(range(len(meta))))

    def test_duplicate_or_singleton_labels_fail(self):
        with self.assertRaises(ValueError):
            make_splits(pd.concat([self.meta, self.meta.iloc[:1]]))
        singleton = pd.DataFrame({"drug": ["a", "b", "b"], "cell_type": ["B", "B", "T"]})
        with self.assertRaises(ValueError):
            make_splits(singleton)

    def test_perfect_zero_and_reversed_predictions(self):
        meta = self.meta.iloc[:3]
        truth = np.array([[1., -2., 3.], [0., 0., 0.], [1., 2., 3.]])
        perfect = score_predictions(meta, truth, truth, "perfect", 0)
        self.assertEqual(perfect.rmse.sum(), 0)
        self.assertAlmostEqual(perfect.pearson.iloc[0], 1.)
        self.assertTrue(np.isnan(perfect.pearson.iloc[1]))
        self.assertTrue(np.isnan(perfect.top50_overlap.iloc[1]))
        self.assertEqual(perfect.top50_sign_accuracy.iloc[0], 1.)
        zero = score_predictions(meta, truth, np.zeros_like(truth), "zero", 0)
        self.assertTrue(zero.pearson.isna().all())
        self.assertTrue(zero.top50_overlap.isna().all())
        self.assertEqual(zero.top50_sign_accuracy.iloc[0], 0.)
        reversed_ = score_predictions(meta, truth, -truth, "reversed", 0)
        self.assertAlmostEqual(reversed_.pearson.iloc[0], -1.)
        self.assertEqual(reversed_.top50_sign_accuracy.iloc[0], 0.)

    def test_score_alignment_follows_metadata_and_array_order(self):
        meta = self.meta.iloc[:3].iloc[[2, 0, 1]]
        truth = np.array([[1., -1.], [2., -2.], [3., -3.]])
        pred = truth + np.array([[1.], [2.], [3.]])
        result = score_predictions(meta, truth, pred, "test", 5, ["gene_a", "gene_b"])
        self.assertEqual(result.cell_type.tolist(), meta.cell_type.tolist())
        np.testing.assert_allclose(result.rmse, [1., 2., 3.])
        with self.assertRaises(ValueError):
            score_predictions(meta, truth, pred, "test", 5, ["gene_a", "gene_a"])

    def test_top_gene_recovery_detects_disjoint_effects(self):
        truth = np.concatenate([np.arange(1., 51.), np.zeros(50)])[None, :]
        wrong_genes = np.concatenate([np.zeros(50), np.arange(1., 51.)])[None, :]
        result = score_predictions(self.meta.iloc[:1], truth, wrong_genes, "wrong", 0)
        self.assertEqual(result.top50_overlap.iloc[0], 0.)
        self.assertEqual(result.top50_sign_accuracy.iloc[0], 0.)

    def test_bootstrap_pairs_labels_after_shuffle(self):
        base = self.meta.copy().assign(model="baseline", fold=0, rmse=2.)
        challenger = self.meta.copy().assign(model="model", fold=0, rmse=1.)
        scores = pd.concat([base, challenger.sample(frac=1, random_state=2)])
        result = bootstrap_rmse_improvement(scores, "baseline", "model", n_boot=100)
        self.assertEqual(result["mean_rmse_improvement"], 1.)
        self.assertEqual(result["ci_low"], 1.)
        self.assertEqual(result["ci_high"], 1.)
        self.assertEqual(result["n_drugs"], 8)
        with self.assertRaises(ValueError):
            bootstrap_rmse_improvement(scores.iloc[:-1], "baseline", "model")


if __name__ == "__main__":
    unittest.main()
