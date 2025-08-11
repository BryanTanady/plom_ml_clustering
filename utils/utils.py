import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score


def get_best_clustering(
    X: np.ndarray, thresholds: list[float], metric="silhouette"
) -> np.ndarray:
    """Get the best clustering of X by searching for optimal threshold that maximizes the metric.

    This function defaults with AgglomerativeClustering clustering algorithm.

    Args:
        X: the feature matrix.
        thresholds: the choices of distance thresholds.
        metric: which metric to optimize. Currently supports: "silhouette" and "davies".

    Returns:
        A numpy array of clusterId where the order matches with the
        inputs (index 0 provides Id for row 0 of X)
    """
    best_score = -np.inf if metric == "silhouette" else np.inf
    best_labels = np.array([])

    for t in thresholds:
        clustering = AgglomerativeClustering(
            n_clusters=None, metric="euclidean", distance_threshold=t
        )
        labels = clustering.fit_predict(X)
        # need at least 2 clusters to score
        if len(set(labels)) < 2:
            continue

        if metric == "silhouette":
            score = silhouette_score(X, labels)
            # silhouette: higher -> better
            if score > best_score:
                best_score = score
                best_labels = labels

        elif metric == "davies":
            score = davies_bouldin_score(X, labels)
            # DB index: lower -> better
            if score < best_score:
                best_score = score
                best_labels = labels

    return best_labels
