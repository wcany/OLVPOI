import numpy as np
import pandas as pd
import scipy.sparse as ss
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


TARGET_COLUMN = "Predicted value"
EXCLUDED_FEATURES = [
    "Anesthesia Time (min)",
    "Surgery Time (min)",
    "Oxygenation Index (OLV-30min)",
    "PaO2 (OLV-30min)",
    "FiO2 (OLV-30min)",
]
BINARY_FEATURES = ["Is_male", "If_left", "Preoperative X or CT"]


def normalize_feature_name(value):
    return " ".join(str(value).strip().split())


def load_feature_table(path):
    frame = pd.read_excel(path)
    column_names = [normalize_feature_name(value) for value in frame.iloc[0]]
    if column_names[-1] != TARGET_COLUMN:
        raise ValueError(
            f"Expected the last column to be {TARGET_COLUMN!r}, got "
            f"{column_names[-1]!r}."
        )
    features = frame.iloc[1:, :-1].copy()
    features.columns = column_names[:-1]
    features = features.apply(pd.to_numeric, errors="raise").reset_index(drop=True)
    labels = pd.to_numeric(frame.iloc[1:, -1], errors="raise").to_numpy(
        dtype=np.int64
    )
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Predicted value must be a binary 0/1 label.")
    missing = [name for name in EXCLUDED_FEATURES if name not in features.columns]
    if missing:
        raise ValueError(
            "Required exclusion columns were not found: " + ", ".join(missing)
        )
    return features.drop(columns=EXCLUDED_FEATURES), labels


def build_feature_schema(features):
    missing = [name for name in BINARY_FEATURES if name not in features.columns]
    if missing:
        raise ValueError("Required binary columns were not found: " + ", ".join(missing))
    for name in BINARY_FEATURES:
        values = set(features[name].dropna().astype(float).unique())
        if not values.issubset({0.0, 1.0}):
            raise ValueError(f"Binary feature {name!r} contains values outside 0/1.")
    return {
        "feature_names": features.columns.tolist(),
        "continuous_features": [
            name for name in features.columns if name not in BINARY_FEATURES
        ],
        "binary_features": list(BINARY_FEATURES),
    }


class FeaturePreprocessor:
    def __init__(self, schema):
        self.schema = schema
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, features):
        self.scaler.fit(features[self.schema["continuous_features"]])
        self.fitted = True
        return self

    def transform(self, features):
        if not self.fitted:
            raise RuntimeError("FeaturePreprocessor must be fitted before transform.")
        transformed = features[self.schema["feature_names"]].astype(float).copy()
        transformed.loc[:, self.schema["continuous_features"]] = self.scaler.transform(
            transformed[self.schema["continuous_features"]]
        )
        binary = transformed[self.schema["binary_features"]].to_numpy(float)
        if not np.isin(binary, [0.0, 1.0]).all():
            raise ValueError("Binary features changed outside 0/1 during preprocessing.")
        return transformed.to_numpy(dtype=np.float32)


def create_fold_assignments(labels, n_splits, seed):
    assignments = np.empty(len(labels), dtype=int)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, test_indices) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
        assignments[test_indices] = fold
    return assignments


def construct_training_adjacency(
    features,
    k,
    algorithm,
    metric,
    minkowski_p,
    minimum_common_neighbors,
):
    """Construct the Step1 pruned KNN mask from the current training data only."""
    if k >= len(features):
        raise ValueError("k must be smaller than the training sample count.")
    neighbors = NearestNeighbors(
        n_neighbors=k + 1,
        algorithm=algorithm,
        metric=metric,
        p=minkowski_p,
    ).fit(features)
    adjacency = neighbors.kneighbors_graph(features)
    adjacency = ss.coo_matrix(adjacency)
    adjacency = (
        adjacency
        + adjacency.T.multiply(adjacency.T > adjacency)
        - adjacency.multiply(adjacency.T > adjacency)
    )
    adjacency = adjacency - ss.dia_matrix(
        (adjacency.diagonal()[np.newaxis, :], [0]), shape=adjacency.shape
    )
    adjacency.eliminate_zeros()

    dense = adjacency.toarray()
    rows, columns = np.nonzero(dense)
    neighbor_sets = {
        row: set(columns[rows == row].tolist()) for row in np.unique(rows)
    }
    for row, column in zip(rows, columns):
        if (
            len(neighbor_sets[row].intersection(neighbor_sets[column]))
            < minimum_common_neighbors
        ):
            dense[row, column] = 0
    adjacency = ss.coo_matrix(dense)
    adjacency.eliminate_zeros()
    return adjacency
