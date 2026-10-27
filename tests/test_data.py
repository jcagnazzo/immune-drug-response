import numpy as np
import pandas as pd
import pytest
from immune_response.data import assert_transfer, load_profiles


def frame(pairs):
    return pd.DataFrame(pairs, columns=["drug", "cell_type"])


def test_transfer_rejects_overlap_and_unknown_compounds():
    train = frame([("a", "T"), ("a", "B"), ("b", "T")])
    assert_transfer(train, frame([("b", "B")]))
    with pytest.raises(ValueError, match="appears in training"):
        assert_transfer(train, frame([("a", "B")]))
    with pytest.raises(ValueError, match="Unseen drug"):
        assert_transfer(train, frame([("c", "B")]))


def test_load_preserves_categorical_order_and_gene_alignment(tmp_path):
    import h5py
    path = tmp_path / "example.h5ad"
    with h5py.File(path, "w") as f:
        for name, categories, codes in [("sm_name", [b"b", b"a"], [1, 0]),
                                         ("cell_type", [b"T", b"B"], [0, 1]),
                                         ("split", [b"train"], [0, 0])]:
            f.create_dataset(f"obs/{name}/categories", data=categories)
            f.create_dataset(f"obs/{name}/codes", data=codes)
        f.create_dataset("layers/logFC", data=[[1, 2], [3, 4]])
        f.create_dataset("var/_index", data=[b"G2", b"G1"])
    meta, response, genes = load_profiles(path)
    assert meta.drug.tolist() == ["a", "b"]
    assert genes.tolist() == ["G2", "G1"]
    np.testing.assert_array_equal(response, [[1, 2], [3, 4]])
