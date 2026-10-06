"""Pairwise-masking Secure Aggregation (Bonawitz et al., 2017), HBC server.

Simulation of the protocol used in the paper:

1. Every unordered client pair (i, j) runs a Diffie-Hellman key agreement
   (RFC 3526 2048-bit MODP group) and hashes the shared secret into a seed
   s_ij.
2. Each seed is expanded by a PRG (NumPy PCG64) into a mask r_ij.
3. Client i sends  u~_i = u_i + sum_{j>i} r_ij - sum_{j<i} r_ji  (Eq. mask).
4. The server sums the masked vectors; all masks cancel (Proposition 1).

Exact cancellation requires modular integer arithmetic, so updates are
encoded in fixed point (``frac_bits`` fractional bits) in Z_{2^32}. The
decoded aggregate differs from the float sum by at most
num_clients * 2^-(frac_bits+1) per coordinate (quantization only); the masks
themselves cancel exactly. Dropout recovery (secret sharing of seeds) is out
of scope, as in the paper (all clients participate in every round).

Not addressed (see the paper's threat model): malicious server, Byzantine
clients, inference from the final model.
"""

from __future__ import annotations

import hashlib
import secrets
from typing import Dict, List, Sequence, Tuple

import numpy as np

# RFC 3526, group 14 (2048-bit MODP), generator 2
_P = int(
    "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74"
    "020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F1437"
    "4FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED"
    "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF05"
    "98DA48361C55D39A69163FA8FD24CF5F83655D23DCA3AD961C62F356208552BB"
    "9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3B"
    "E39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF695581718"
    "3995497CEA956AE515D2261898FA051015728E5A8AACAA68FFFFFFFFFFFFFFFF", 16)
_G = 2
_MOD_BITS = 32
_MASK = np.uint64((1 << _MOD_BITS) - 1)


class DHParty:
    def __init__(self, rng_bytes: bytes | None = None):
        self._sk = int.from_bytes(rng_bytes or secrets.token_bytes(32), "big")
        self.pk = pow(_G, self._sk, _P)

    def shared_seed(self, other_pk: int) -> int:
        secret = pow(other_pk, self._sk, _P)
        digest = hashlib.sha256(secret.to_bytes(256, "big")).digest()
        return int.from_bytes(digest[:8], "big")


def setup_pairwise_seeds(num_clients: int, seed: int | None = None) -> Dict[Tuple[int, int], int]:
    """Run DH between all pairs; returns {(i, j): s_ij} for i < j.

    ``seed`` makes the secret keys deterministic (for reproducible experiments
    and tests only; a deployment must use fresh randomness).
    """
    if seed is None:
        parties = [DHParty() for _ in range(num_clients)]
    else:
        g = np.random.default_rng(seed)
        parties = [DHParty(g.bytes(32)) for _ in range(num_clients)]
    seeds = {}
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            s_ij = parties[i].shared_seed(parties[j].pk)
            assert s_ij == parties[j].shared_seed(parties[i].pk)
            seeds[(i, j)] = s_ij
    return seeds


def _prg(seed: int, round_idx: int, dim: int, stream: int = 0) -> np.ndarray:
    """Mask r_ij for one (round, parameter-tensor) stream."""
    g = np.random.Generator(np.random.PCG64([seed, round_idx, stream]))
    return g.integers(0, 1 << _MOD_BITS, size=dim, dtype=np.uint64)


def encode(x: np.ndarray, frac_bits: int = 24) -> np.ndarray:
    q = np.round(np.asarray(x, dtype=np.float64) * (1 << frac_bits)).astype(np.int64)
    return q.astype(np.uint64) & _MASK


def decode(v: np.ndarray, frac_bits: int = 24) -> np.ndarray:
    v = v.astype(np.uint64) & _MASK
    signed = v.astype(np.int64)
    signed = np.where(signed >= (1 << (_MOD_BITS - 1)), signed - (1 << _MOD_BITS), signed)
    return signed.astype(np.float64) / (1 << frac_bits)


def mask_update(client_id: int, encoded: np.ndarray, seeds: Dict[Tuple[int, int], int],
                num_clients: int, round_idx: int, stream: int = 0) -> np.ndarray:
    m = np.zeros_like(encoded, dtype=np.uint64)
    for j in range(num_clients):
        if j == client_id:
            continue
        a, b = min(client_id, j), max(client_id, j)
        r = _prg(seeds[(a, b)], round_idx, encoded.size, stream)
        m = (m + r) & _MASK if client_id < j else (m - r) & _MASK
    return (encoded + m) & _MASK


def aggregate_masked(masked: Sequence[np.ndarray]) -> np.ndarray:
    total = np.zeros_like(masked[0], dtype=np.uint64)
    for v in masked:
        total = (total + v) & _MASK
    return total


def secure_sum(vectors: List[np.ndarray], round_idx: int = 0,
               seeds: Dict[Tuple[int, int], int] | None = None,
               frac_bits: int = 24) -> Tuple[np.ndarray, List[np.ndarray]]:
    """End-to-end helper: returns (decoded aggregate, list of masked messages)."""
    k = len(vectors)
    seeds = seeds or setup_pairwise_seeds(k)
    masked = [mask_update(i, encode(v, frac_bits), seeds, k, round_idx)
              for i, v in enumerate(vectors)]
    return decode(aggregate_masked(masked), frac_bits), masked


def payload_bytes(dim: int) -> int:
    """Bytes per masked message in this implementation (uint32 coordinates)."""
    return dim * (_MOD_BITS // 8)
