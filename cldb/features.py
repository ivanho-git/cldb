import hashlib
import numpy as np
from cldb.parser import QueryFeatures


class FeatureEncoder:
    def __init__(self, max_tables: int = 20, max_cols: int = 50):
        self.max_tables = max_tables
        self.max_cols = max_cols

    def _hash_bucket(self, value: str, num_buckets: int) -> int:
        m = hashlib.md5(value.lower().encode("utf-8"), usedforsecurity=False)
        return int(m.hexdigest(), 16) % num_buckets

    def encode(self, features: QueryFeatures) -> np.ndarray:
        vec = []

        vec.append(float(features.join_count))
        vec.append(1.0 if features.has_group_by else 0.0)
        vec.append(1.0 if features.has_order_by else 0.0)

        card = features.estimated_cardinality or 0.0
        vec.append(np.log1p(card) if card > 0 else 0.0)

        table_vec = [0.0] * self.max_tables
        for t in features.tables:
            idx = self._hash_bucket(t, self.max_tables)
            table_vec[idx] = 1.0
        vec.extend(table_vec)

        col_vec = [0.0] * self.max_cols
        for c in features.predicate_columns:
            idx = self._hash_bucket(c, self.max_cols)
            col_vec[idx] = 1.0
        vec.extend(col_vec)

        return np.array(vec, dtype=np.float32)

    def get_feature_dim(self) -> int:
        return 4 + self.max_tables + self.max_cols