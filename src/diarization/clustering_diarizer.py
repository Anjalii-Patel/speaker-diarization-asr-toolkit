"""
Clustering-Based Speaker Diarizer
Applies Agglomerative Hierarchical Clustering, Spectral Clustering, or DBSCAN to speaker embeddings.
Provides detailed diagnostics: pairwise cosine similarity matrix, cluster distribution, and outlier logging.
"""

import logging
from typing import List, Optional, Tuple, Union, Dict, Any
import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
from .rttm import RTTMSegment

logger = logging.getLogger(__name__)

class ClusteringDiarizer:
    """
    Partitions audio segments into speaker identities using clustering algorithms.
    """

    def __init__(
        self,
        method: str = "agglomerative",
        num_speakers: Optional[int] = 2,
        linkage: str = "ward",
        metric: str = "cosine",
        distance_threshold: Optional[float] = None,
        dbscan_eps: float = 0.5,
        dbscan_min_samples: int = 1,
    ):
        self.method = method.lower()
        self.num_speakers = num_speakers
        self.linkage = linkage.lower()
        self.metric = metric.lower()
        self.distance_threshold = distance_threshold
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples

        # Inspection attributes populated after fit_predict
        self.similarity_matrix_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Cluster embedding vectors into speaker labels.

        Args:
            embeddings: 2D numpy array of shape (num_segments, feature_dim).

        Returns:
            1D numpy array of integer speaker cluster labels.
        """
        num_samples = len(embeddings)
        if num_samples == 0:
            self.similarity_matrix_ = np.empty((0, 0))
            self.labels_ = np.array([], dtype=int)
            return self.labels_

        if num_samples == 1:
            self.similarity_matrix_ = np.array([[1.0]])
            self.labels_ = np.array([0], dtype=int)
            return self.labels_

        # Calculate and log pairwise cosine similarity matrix diagnostics
        sim_matrix = cosine_similarity(embeddings)
        self.similarity_matrix_ = sim_matrix

        # Off-diagonal similarities
        mask = ~np.eye(num_samples, dtype=bool)
        off_diag = sim_matrix[mask] if num_samples > 1 else np.array([1.0])
        sim_min = float(np.min(off_diag))
        sim_max = float(np.max(off_diag))
        sim_mean = float(np.mean(off_diag))
        logger.info(
            "Pairwise cosine similarity (%d segments): min=%.3f, mean=%.3f, max=%.3f",
            num_samples, sim_min, sim_mean, sim_max
        )

        if self.method == "agglomerative":
            # For L2-normalized embeddings, Euclidean distance is monotonically related
            # to cosine distance: ||u - v||^2 = 2 - 2 * cos(u, v).
            # Ward linkage requires euclidean metric and minimizes within-cluster spherical variance.
            if self.linkage == "ward":
                actual_metric = "euclidean"
            else:
                actual_metric = self.metric

            actual_clusters = self.num_speakers
            if actual_clusters is not None:
                actual_clusters = min(actual_clusters, num_samples)

            # If distance threshold is provided, n_clusters must be None
            if self.distance_threshold is not None:
                actual_clusters = None

            clusterer = AgglomerativeClustering(
                n_clusters=actual_clusters,
                distance_threshold=self.distance_threshold,
                metric=actual_metric,
                linkage=self.linkage,
            )
            labels = clusterer.fit_predict(embeddings)

        elif self.method == "spectral":
            actual_clusters = self.num_speakers or 2
            actual_clusters = min(actual_clusters, num_samples)
            # Affinity derived from non-negative cosine similarity
            affinity_matrix = np.clip(sim_matrix, 0.0, 1.0)
            np.fill_diagonal(affinity_matrix, 1.0)
            clusterer = SpectralClustering(
                n_clusters=actual_clusters,
                affinity="precomputed",
                random_state=42,
            )
            labels = clusterer.fit_predict(affinity_matrix)

        elif self.method == "dbscan":
            actual_metric = "cosine"if self.metric == "cosine"else "euclidean"
            clusterer = DBSCAN(
                eps=self.dbscan_eps,
                min_samples=self.dbscan_min_samples,
                metric=actual_metric,
            )
            labels = clusterer.fit_predict(embeddings)
            # Re-map any -1 (noise) to the highest cluster ID or 0
            if -1 in labels:
                max_label = max([l for l in labels if l != -1], default=-1)
                labels = np.array([l if l != -1 else max_label + 1 for l in labels])

        else:
            raise ValueError(f"Unsupported clustering method: {self.method}")

        self.labels_ = labels

        # Cluster distribution logging
        unique, counts = np.unique(labels, return_counts=True)
        dist_str = ", ".join(f"SPEAKER_{u:02d}: {c}"for u, c in zip(unique, counts))
        logger.info("Cluster distribution across %d segments: %s", num_samples, dist_str)

        # Check for extreme cluster imbalance (e.g. outlier isolation)
        if len(unique) > 1 and min(counts) == 1 and max(counts) >= (num_samples - 1):
            logger.warning(
                "Warning: Detected potential singleton outlier cluster (counts: %s). "
                "Consider using linkage='ward' or sub-segmenting long utterances.",
                dict(zip(unique.tolist(), counts.tolist()))
            )

        return labels

    def assign_speaker_labels(
        self,
        speech_intervals: List[Tuple[float, float]],
        cluster_labels: np.ndarray,
        uri: str = "audio",
    ) -> List[RTTMSegment]:
        """
        Format timestamps and cluster labels into RTTMSegment objects.

        Args:
            speech_intervals: List of (start_seconds, end_seconds) tuples.
            cluster_labels: Array of cluster IDs.
            uri: Audio recording identifier.

        Returns:
            List of RTTMSegment instances.
        """
        if len(speech_intervals) != len(cluster_labels):
            raise ValueError(
                f"Mismatch: {len(speech_intervals)} intervals vs {len(cluster_labels)} labels."
            )

        rttm_segments: List[RTTMSegment] = []
        for (start, end), label in zip(speech_intervals, cluster_labels):
            speaker_id = f"SPEAKER_{int(label):02d}"
            duration = max(0.01, end - start)
            rttm_segments.append(
                RTTMSegment(
                    uri=uri,
                    channel=1,
                    start=round(start, 3),
                    duration=round(duration, 3),
                    speaker=speaker_id,
                )
            )

        rttm_segments.sort(key=lambda s: s.start)
        return rttm_segments
