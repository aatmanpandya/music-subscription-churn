"""One shared preprocessing path: identical at training and scoring time."""
import numpy as np
import pandas as pd
import pandas.testing as pdt

from pipeline.features import Preprocessor
from pipeline.schema import CAT, FEATURES, NUM
from tests.conftest import make_frame


def test_matrix_shape_dtypes_and_order(train_frame):
    prep = Preprocessor.fit(train_frame)
    X = prep.transform(train_frame)
    assert list(X.columns) == FEATURES
    assert all(X[c].dtype == "float64" for c in NUM)
    assert all(isinstance(X[c].dtype, pd.CategoricalDtype) for c in CAT)


def test_scoring_rows_get_the_same_features_as_training_rows(train_frame):
    """Transforming a subset in isolation (as a scoring batch would be) must equal
    the same rows from the full training transform."""
    prep = Preprocessor.fit(train_frame)
    full = prep.transform(train_frame)
    subset = train_frame.sample(500, random_state=3)
    scored = prep.transform(subset)
    pdt.assert_frame_equal(scored, full.loc[subset.index])


def test_saved_preprocessor_roundtrip_is_identical(train_frame, tmp_path):
    prep = Preprocessor.fit(train_frame)
    prep.save(tmp_path / "p.json")
    again = Preprocessor.load(tmp_path / "p.json")
    pdt.assert_frame_equal(prep.transform(train_frame), again.transform(train_frame))


def test_categories_are_pinned_not_rederived_from_the_batch(train_frame):
    prep = Preprocessor.fit(train_frame)
    # A batch containing only one city must still carry the full training vocabulary.
    one_city = train_frame[train_frame.city == 5].head(200)
    X = prep.transform(one_city)
    assert list(X["city"].cat.categories) == prep.categories["city"]
    assert (X["city"].cat.codes == prep.categories["city"].index("5")).all()


def test_unseen_category_becomes_missing_and_is_counted(train_frame):
    prep = Preprocessor.fit(train_frame)
    row = train_frame.head(10).copy()
    row["city"] = 99.0
    X = prep.transform(row)
    assert X["city"].isna().all()
    assert prep.unseen_share(row)["city"] == 1.0


def test_dtype_representation_does_not_change_the_matrix(train_frame):
    """Int64 (DuckDB parquet) vs float64 (notebook) inputs must give identical codes."""
    prep = Preprocessor.fit(train_frame)
    alt = train_frame.copy()
    for c in ["city", "registered_via", "payment_method", "plan_days"]:
        alt[c] = alt[c].astype("Int64")
    pdt.assert_frame_equal(prep.transform(train_frame), prep.transform(alt))


def test_scorer_uses_the_same_path_as_training(trained, train_frame):
    scorer = trained["scorer"]
    X = scorer.prep.transform(train_frame.head(300))
    direct = scorer.raw_proba_from_matrix(X)
    via_predict = scorer.predict(train_frame.head(300), validate=False)["raw_probability"].to_numpy()
    np.testing.assert_allclose(direct, via_predict)


def test_mismatched_feature_list_is_refused(train_frame):
    d = Preprocessor.fit(train_frame).to_dict()
    d["numeric"] = d["numeric"][:-1]
    import pytest
    with pytest.raises(ValueError, match="does not match"):
        Preprocessor.from_dict(d)
