import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

from strable_leaderboard.encoders import ENCODERS, N_DIMS, NAMES, encode, string_columns

WORDS = [f"w{i}" for i in range(60)]


def _cache(tmp_path, encoder, dims, table="t"):
    """A filled embedding cache, so that no language model runs."""
    rng = np.random.default_rng(0)
    emb = pd.DataFrame(rng.normal(size=(len(WORDS) + 1, dims)).astype("float32"), columns=[f"X{i}" for i in range(dims)])
    emb.insert(0, "name", WORDS + ["nan"])
    folder = tmp_path / "embeddings" / encoder
    folder.mkdir(parents=True)
    emb.to_parquet(folder / f"{table}.parquet", index=False)
    (folder / f"{table}.time").write_text("2.5")
    return emb.set_index("name")


def _frames(n_rare=10):
    rng = np.random.default_rng(1)
    many = pd.Series(rng.choice(WORDS, 80), dtype=object)
    many[::7] = np.nan
    rare = pd.Series([np.nan] * 80, dtype=object)
    rare[:n_rare] = rng.choice(WORDS, n_rare)  # fewer non-missing cells than components
    X = pd.DataFrame({"many": many, "rare": rare, "x": rng.normal(size=80)})
    return X.iloc[:60], X.iloc[60:]


def test_four_encoders_named_like_the_paper():
    assert list(ENCODERS) == ["tfidf", "qwen-3-8b-first30", "llama-3.1-8b-first30", "minilm-l6-v2-pca30"]
    assert set(NAMES) == set(ENCODERS) | {"none"}
    assert all(spec["revision"] for spec in ENCODERS.values() if spec["checkpoint"])


def test_first30_slices_and_keeps_missing(tmp_path):
    emb = _cache(tmp_path, "qwen-3-8b-first30", N_DIMS)
    X_train, X_test = _frames()
    tr, te, seconds = encode(X_train, X_test, "qwen-3-8b-first30", "t", tmp_path)
    assert seconds >= 2.5 and tr.shape == (60, 2 * N_DIMS + 1) and te.shape == (20, 2 * N_DIMS + 1)
    many = tr[[f"many_{i}" for i in range(N_DIMS)]].to_numpy()
    seen = X_train["many"].notna().to_numpy()
    assert np.isnan(many[~seen]).all()
    np.testing.assert_array_equal(many[seen], emb.loc[X_train["many"][seen]].to_numpy()[:, :N_DIMS])


def test_pca30_is_fitted_on_non_missing_training_cells(tmp_path):
    emb = _cache(tmp_path, "minilm-l6-v2-pca30", 64)
    X_train, X_test = _frames()
    tr, te, _ = encode(X_train, X_test, "minilm-l6-v2-pca30", "t", tmp_path)
    seen_tr, seen_te = X_train["many"].notna().to_numpy(), X_test["many"].notna().to_numpy()
    pca = PCA(n_components=N_DIMS, random_state=1234)
    expected_tr = pca.fit_transform(emb.loc[X_train["many"][seen_tr]].to_numpy())
    cols = [f"many_{i}" for i in range(N_DIMS)]
    np.testing.assert_allclose(tr[cols].to_numpy()[seen_tr], expected_tr, atol=1e-3)
    np.testing.assert_allclose(te[cols].to_numpy()[seen_te], pca.transform(emb.loc[X_test["many"][seen_te]].to_numpy()),
                               atol=1e-3)
    assert np.isnan(tr[cols].to_numpy()[~seen_tr]).all() and np.isnan(te[cols].to_numpy()[~seen_te]).all()
    # 10 training cells < 30 components: the first 30 dimensions instead, as in the paper
    rare = tr[[f"rare_{i}" for i in range(N_DIMS)]].to_numpy()[:10]
    np.testing.assert_array_equal(rare, emb.loc[X_train["rare"][:10]].to_numpy()[:, :N_DIMS])


@pytest.mark.parametrize("encoder", ["tfidf", "llama-3.1-8b-first30", "minilm-l6-v2-pca30"])
def test_dates_are_not_strings(tmp_path, encoder):
    if encoder != "tfidf":
        _cache(tmp_path, encoder, 64)
    X_train, X_test = _frames()
    X_train = X_train.assign(day=pd.date_range("2020-01-01", periods=60).astype(str))
    X_test = X_test.assign(day=pd.date_range("2021-01-01", periods=20).astype(str))
    assert string_columns(X_train) == ["many", "rare"]
    tr, te, _ = encode(X_train, X_test, encoder, "t", tmp_path)
    assert {"day_year", "day_month"} <= set(tr.columns) and tr["day_year"].abs().max() <= 3  # squashed, as in the paper
