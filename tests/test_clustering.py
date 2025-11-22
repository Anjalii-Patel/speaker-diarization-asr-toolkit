"""Tests for clustering diarization algorithms."""

import numpy as np
import pytest
from src.diarization.clustering_diarizer import ClusteringDiarizer

def test_agglomerative_clustering_two_distinct_clusters():
    # 3 samples around [1, 1], 3 samples around [-1, -1]
    cluster_a = np.array([[1.0, 1.0], [1.1, 0.9], [0.9, 1.1]])
    cluster_b = np.array([[-1.0, -1.0], [-1.1, -0.9], [-0.9, -1.1]])
    embeddings = np.vstack([cluster_a, cluster_b])

    diarizer = ClusteringDiarizer(method="agglomerative", num_speakers=2, linkage="average", metric="euclidean")
    labels = diarizer.fit_predict(embeddings)

    assert len(labels) == 6
    # Samples in cluster_a should share label, and differ from cluster_b
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]

def test_dbscan_clustering():
    cluster_a = np.array([[0.0, 0.0], [0.05, 0.05], [-0.05, -0.05]])
    cluster_b = np.array([[10.0, 10.0], [10.05, 10.05], [9.95, 9.95]])
    embeddings = np.vstack([cluster_a, cluster_b])

    diarizer = ClusteringDiarizer(method="dbscan", dbscan_eps=0.5, dbscan_min_samples=2, metric="euclidean")
    labels = diarizer.fit_predict(embeddings)

    assert len(labels) == 6
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]

def test_edge_cases():
    diarizer = ClusteringDiarizer()
    # Empty
    empty_labels = diarizer.fit_predict(np.empty((0, 2)))
    assert len(empty_labels) == 0

    # Single segment
    single_label = diarizer.fit_predict(np.array([[1.0, 2.0]]))
    assert len(single_label) == 1
    assert single_label[0] == 0

def test_assign_speaker_labels():
    diarizer = ClusteringDiarizer()
    intervals = [(0.0, 1.5), (2.0, 3.5)]
    labels = np.array([0, 1])
    rttm_segs = diarizer.assign_speaker_labels(intervals, labels, uri="test")

    assert len(rttm_segs) == 2
    assert rttm_segs[0].speaker == "SPEAKER_00"
    assert rttm_segs[0].start == 0.0
    assert rttm_segs[0].duration == 1.5
    assert rttm_segs[1].speaker == "SPEAKER_01"
