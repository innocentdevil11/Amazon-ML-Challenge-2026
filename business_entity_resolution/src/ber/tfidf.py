"""Char n-gram TF-IDF used to (a) re-rank the blocking union down to
``k_final`` candidates per S1, and (b) provide a name/address cosine feature
to the matching model.

A ``HashingVectorizer`` is used instead of a vocabulary-based
``TfidfVectorizer`` because the corpus is O(10M) documents across three
scripts/languages worth of tokens after transliteration -- hashing avoids
building and pickling a multi-million-entry vocabulary. IDF weights (which a
pure hashing vectorizer doesn't have) are fit separately with a
``TfidfTransformer`` on a bounded sample, and reused as-is at test time so
the test split's document frequencies never leak into weighting.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

N_FEATURES = 2**20


def build_hashing_vectorizer(n_features: int = N_FEATURES) -> HashingVectorizer:
    return HashingVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 4),
        n_features=n_features,
        alternate_sign=False,
        norm=None,
        lowercase=False,  # text is already normalized to lowercase
    )


def fit_idf(vectorizer: HashingVectorizer, sample_texts: list[str], chunk_size: int = 200_000) -> TfidfTransformer:
    """Fit IDF weights on ``sample_texts``, transforming in chunks.

    ``HashingVectorizer.transform`` allocates one internal buffer sized for
    its *entire* input batch; at full-dataset scale the train split's
    IDF-fitting sample is ~5.2M documents, and that single-call buffer
    failed a 4GiB allocation under real memory pressure (confirmed on the
    full run: ``numpy._core._exceptions._ArrayMemoryError`` inside
    ``HashingVectorizer.transform``, from a single unchunked call on the
    complete sample). Transforming in bounded chunks and stacking keeps
    every individual allocation small regardless of total sample size.
    """
    if len(sample_texts) <= chunk_size:
        counts = vectorizer.transform(sample_texts)
    else:
        parts = [
            vectorizer.transform(sample_texts[i : i + chunk_size])
            for i in range(0, len(sample_texts), chunk_size)
        ]
        counts = sp.vstack(parts, format="csr")
    transformer = TfidfTransformer(sublinear_tf=True, norm="l2")
    transformer.fit(counts)
    return transformer


def vectorize(vectorizer: HashingVectorizer, idf: TfidfTransformer, texts: list[str]) -> sp.csr_matrix:
    counts = vectorizer.transform(texts)
    return idf.transform(counts)


def save_idf(idf: TfidfTransformer, path: str | Path) -> None:
    Path(path).write_bytes(pickle.dumps(idf))


def load_idf(path: str | Path) -> TfidfTransformer:
    return pickle.loads(Path(path).read_bytes())


def pair_cosine(mat_a: sp.csr_matrix, mat_b: sp.csr_matrix) -> np.ndarray:
    """Row-aligned cosine similarity: ``mat_a`` and ``mat_b`` must have the
    same number of rows, each row i is one (a_i, b_i) pair. Both must already
    be L2-normalized (true for anything from :func:`vectorize`)."""
    return np.asarray(mat_a.multiply(mat_b).sum(axis=1)).ravel()


def _score_chunk(vectorizer: HashingVectorizer, idf: TfidfTransformer, texts_a: list[str], texts_b: list[str]) -> np.ndarray:
    return pair_cosine(vectorize(vectorizer, idf, texts_a), vectorize(vectorizer, idf, texts_b))


def pair_cosine_chunked(
    vectorizer: HashingVectorizer, idf: TfidfTransformer, texts_a: list[str], texts_b: list[str],
    n_jobs: int = 1, chunk_size: int = 200_000,
) -> np.ndarray:
    """Same as vectorizing both sides and calling :func:`pair_cosine`, but
    chunked across worker processes -- hashing+sparse-cosine over tens of
    millions of pairs is CPU-bound and this is the difference between using
    1 core and all of them."""
    n = len(texts_a)
    if n_jobs <= 1 or n < chunk_size * 2:
        return _score_chunk(vectorizer, idf, texts_a, texts_b)
    from joblib import Parallel, delayed

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_score_chunk)(vectorizer, idf, texts_a[i : i + chunk_size], texts_b[i : i + chunk_size])
        for i in range(0, n, chunk_size)
    )
    return np.concatenate(results)
