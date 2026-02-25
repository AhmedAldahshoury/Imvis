import pytest

np = pytest.importorskip("numpy")

from app.indexer import kmeans_labels, project_vectors_2d


def test_project_vectors_2d_shape_and_scale():
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    projected = project_vectors_2d(vectors)

    assert projected.shape == (3, 2)
    assert np.max(np.abs(projected)) <= 1.0 + 1e-6


def test_kmeans_labels_returns_expected_groups():
    points = np.array(
        [
            [-1.0, -1.0],
            [-0.9, -0.8],
            [1.0, 1.0],
            [0.8, 1.1],
        ],
        dtype=np.float32,
    )

    labels = kmeans_labels(points, k=2, iterations=20)

    assert labels.shape == (4,)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]