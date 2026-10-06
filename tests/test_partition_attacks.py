import numpy as np

from fepp_cd.attacks import flip_labels
from fepp_cd.partition import describe_partition, dirichlet_partition, iid_partition


def test_iid_partition_covers_all():
    parts = iid_partition(1003, 5, seed=42)
    allidx = np.concatenate(parts)
    assert len(allidx) == 1003 and len(np.unique(allidx)) == 1003


def test_dirichlet_partition_assigns_every_sample_once_and_is_deterministic():
    y = np.r_[np.zeros(500), np.ones(3000)].astype(int)
    p1 = dirichlet_partition(y, 5, alpha=0.5, seed=42)
    p2 = dirichlet_partition(y, 5, alpha=0.5, seed=42)
    allidx = np.concatenate(p1)
    assert len(allidx) == len(y) and len(np.unique(allidx)) == len(y)
    assert all(np.array_equal(a, b) for a, b in zip(p1, p2))
    d = describe_partition(p1, y)
    assert d.n_total.sum() == len(y)


def test_dirichlet_skew_increases_as_alpha_decreases():
    y = np.r_[np.zeros(5000), np.ones(5000)].astype(int)
    def spread(a):
        d = describe_partition(dirichlet_partition(y, 5, alpha=a, seed=0), y)
        return d.pos_pct.std()
    assert spread(0.1) > spread(100.0)


def test_flip_labels_counts_and_direction():
    y = np.array([1] * 100 + [0] * 50)
    new, idx = flip_labels(y, 0.3, seed=0)
    assert len(idx) == 30 and (y[idx] == 1).all() and (new[idx] == 0).all()
    assert new.sum() == 70 and (new[100:] == 0).all()
    same, none = flip_labels(y, 0.0)
    assert len(none) == 0 and np.array_equal(same, y)
