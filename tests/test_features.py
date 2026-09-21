import pytest
from cldb.parser import QueryFeatures
from cldb.features import FeatureEncoder


def test_feature_encoder_dim():
    encoder = FeatureEncoder(max_tables=20, max_cols=50)
    assert encoder.get_feature_dim() == 74


def test_feature_encoder_scalar_features():
    encoder = FeatureEncoder()
    features = QueryFeatures(
        tables=[],
        join_count=3,
        predicate_columns=[],
        has_group_by=True,
        has_order_by=True,
        estimated_cardinality=100.0,
    )
    vec = encoder.encode(features)
    assert vec[0] == 3.0
    assert vec[1] == 1.0
    assert vec[2] == 1.0
    assert vec[3] > 0.0


def test_feature_encoder_tables_bow():
    encoder = FeatureEncoder()
    features = QueryFeatures(
        tables=["users", "orders"],
        join_count=0,
        predicate_columns=[],
        has_group_by=False,
        has_order_by=False,
    )
    vec = encoder.encode(features)
    assert vec.shape[0] == 74
    assert sum(vec[4 : 4 + encoder.max_tables]) == 2.0


def test_feature_encoder_columns_bow():
    encoder = FeatureEncoder()
    features = QueryFeatures(
        tables=[],
        join_count=0,
        predicate_columns=["status", "created_at"],
        has_group_by=False,
        has_order_by=False,
    )
    vec = encoder.encode(features)
    assert sum(vec[4 + encoder.max_tables :]) == 2.0


def test_feature_encoder_deterministic():
    encoder = FeatureEncoder()
    features1 = QueryFeatures(
        tables=["orders"],
        join_count=1,
        predicate_columns=["customer_id"],
        has_group_by=False,
        has_order_by=False,
    )
    features2 = QueryFeatures(
        tables=["orders"],
        join_count=1,
        predicate_columns=["customer_id"],
        has_group_by=False,
        has_order_by=False,
    )
    vec1 = encoder.encode(features1)
    vec2 = encoder.encode(features2)
    assert (vec1 == vec2).all()


def test_feature_encoder_empty():
    encoder = FeatureEncoder()
    features = QueryFeatures()
    vec = encoder.encode(features)
    assert vec.shape[0] == 74
    assert vec[0] == 0.0
    assert vec[1] == 0.0
    assert vec[2] == 0.0
    assert vec[3] == 0.0
    assert vec[4:].sum() == 0.0