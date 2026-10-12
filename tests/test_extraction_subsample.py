"""Capture-time position subsampling must keep one fixed subset per sequence
length: every depth of one forward pass shares it, and a later pass of a
different length within the same extraction (TabICL above ten classes runs
the transformer several times) gets its own instead of indexing out of bounds."""

import torch

from shallow_tfm.extraction import _subsample_positions


def test_same_subset_across_depths_and_per_length():
    cache = {}
    positions = torch.arange(1200, dtype=torch.float32)[:, None, None].expand(1200, 5, 8)
    a0 = _subsample_positions(positions.clone(), 500, cache)
    a1 = _subsample_positions(positions.clone(), 500, cache)
    assert a0.shape == a1.shape == (500, 5, 8)
    assert torch.equal(a0, a1)  # one fixed subset for that length, shared across depths
    b = _subsample_positions(torch.randn(700, 5, 8), 500, cache)  # a later, shorter pass
    assert b.shape == (500, 5, 8) and cache[700].max() < 700
    assert _subsample_positions(torch.randn(300, 5, 8), 500, cache).shape == (300, 5, 8)  # below the cap: untouched
