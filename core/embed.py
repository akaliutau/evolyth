from __future__ import annotations

import hashlib
import math

VECTOR_DIM = 64


def embed_text(text: str, dim: int = VECTOR_DIM) -> list[float]:
    """Tiny deterministic embedding to keep the demo dependency-light.

    It is not a semantic model. Replace this function with sentence-transformers
    later; the rest of the code does not need to change.
    """
    v = [0.0] * dim
    for token in (text or "").lower().replace("_", " ").split():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        v[bucket] += sign
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [float(x / norm) for x in v]
