"""Download and validate the public, processed OP3 benchmark matrices."""

from pathlib import Path
import hashlib
import json
import urllib.request

import h5py
import numpy as np
import pandas as pd

BASE = "https://openproblems-bio.s3.amazonaws.com/public/neurips-2023-competition/workflow-resources/neurips-2023-data/"
FILES = {
    "de_train.h5ad": "532d0ffa616233ba571d2300813cffc8bcdd0f83a78de7119cb229df8334e375",
    "de_test.h5ad": "5746d870d61e453b8b404b512dbe310b961f38f1a5b5ca1c67ca243d673e868d",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        target = directory / name
        if target.exists() and sha256(target) == expected:
            continue
        partial = target.with_suffix(".partial")
        print(f"Downloading {name}", flush=True)
        urllib.request.urlretrieve(BASE + name, partial)
        if sha256(partial) != expected:
            partial.unlink()
            raise ValueError(f"Checksum mismatch: {name}")
        partial.replace(target)


def _strings(values):
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in values])


def _column(node):
    if isinstance(node, h5py.Group):
        categories = _strings(node["categories"][:])
        codes = node["codes"][:]
        if np.any(codes < 0):
            raise ValueError("Missing categorical metadata")
        return categories[codes]
    values = node[:]
    return _strings(values) if values.dtype.kind in "OSU" else values


def load_profiles(path, verify=True):
    path = Path(path)
    if verify and path.name in FILES and sha256(path) != FILES[path.name]:
        raise ValueError(f"Unexpected input bytes: {path.name}")
    with h5py.File(path) as f:
        meta = pd.DataFrame({"drug": _column(f["obs/sm_name"]),
                             "cell_type": _column(f["obs/cell_type"]),
                             "split": _column(f["obs/split"])})
        y = f["layers/logFC"][:].astype(np.float32)
        genes = _strings(f["var/_index"][:])
    if y.shape != (len(meta), len(genes)) or not np.isfinite(y).all():
        raise ValueError("Invalid response matrix")
    if meta.duplicated(["drug", "cell_type"]).any() or len(np.unique(genes)) != len(genes):
        raise ValueError("Repeated condition or gene identifiers")
    return meta, y, genes


def assert_transfer(train, test):
    pairs = [set(map(tuple, m[["drug", "cell_type"]].to_numpy())) for m in (train, test)]
    if pairs[0] & pairs[1]:
        raise ValueError("A test drug-cell pair appears in training")
    for column in ("drug", "cell_type"):
        if not set(test[column]) <= set(train[column]):
            raise ValueError(f"Unseen {column}: this study tests known-label combinations")


def write_manifest(directory, output):
    info = {name: {"url": BASE + name, "sha256": sha256(Path(directory) / name)} for name in FILES}
    Path(output).write_text(json.dumps(info, indent=2) + "\n")
