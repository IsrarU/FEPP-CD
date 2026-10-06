import numpy as np

from fepp_cd import secagg


def test_masks_cancel_exactly():
    rng = np.random.default_rng(0)
    vecs = [rng.normal(0, 0.01, 1000) for _ in range(5)]
    seeds = secagg.setup_pairwise_seeds(5, seed=1)
    enc = [secagg.encode(v) for v in vecs]
    masked = [secagg.mask_update(i, e, seeds, 5, round_idx=3) for i, e in enumerate(enc)]
    # Proposition 1: sum of masked == sum of unmasked (exactly, in Z_2^32)
    assert np.array_equal(secagg.aggregate_masked(masked), secagg.aggregate_masked(enc))


def test_aggregate_matches_float_sum_within_quantization():
    rng = np.random.default_rng(1)
    vecs = [rng.normal(0, 0.05, 5000) for _ in range(5)]
    agg, _ = secagg.secure_sum(vecs, seeds=secagg.setup_pairwise_seeds(5, seed=2))
    assert np.max(np.abs(agg - np.sum(vecs, axis=0))) <= 5 * 2.0 ** -25 + 1e-12


def test_individual_messages_do_not_reveal_updates():
    rng = np.random.default_rng(2)
    vecs = [rng.normal(0, 0.01, 2000) for _ in range(3)]
    _, masked = secagg.secure_sum(vecs, seeds=secagg.setup_pairwise_seeds(3, seed=3))
    for v, m in zip(vecs, masked):
        decoded = secagg.decode(m)
        # a masked message decodes to (near-uniform) noise, uncorrelated with the update
        assert abs(np.corrcoef(decoded, v)[0, 1]) < 0.1
        assert np.std(decoded) > 1000 * np.std(v)


def test_dh_seeds_are_symmetric_and_distinct():
    seeds = secagg.setup_pairwise_seeds(4, seed=7)
    assert len(seeds) == 6 and len(set(seeds.values())) == 6
