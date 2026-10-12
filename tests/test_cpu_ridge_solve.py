"""The CPU per-feature ridge fit must survive a re-initialised OpenMP pool.

``torch.linalg.solve`` on a batched CPU input goes through MKL's batched LU,
which can return wrong pivots, print "Parameter 6 was incorrect on entry to
SLASWP" or crash once ``torch.set_num_threads()`` has re-initialised the
OpenMP pool; the library solves per matrix on CPU instead.  The trigger state
is set in a subprocess so a regression cannot take the pytest process down
with it.
"""

import os
import subprocess
import sys
import textwrap

import torch

# TabPFN-like shape: 39 feature tokens x 400 fit positions x d=192.
_SUBPROCESS = textwrap.dedent("""
    import os, torch
    torch.set_num_threads(max(2, os.cpu_count() or 2))   # the trigger state
    from shallow_tfm.translators import fit_per_feature_auto, fit_ridge
    g = torch.Generator().manual_seed(0)
    H_s = torch.randn(400, 39, 192, generator=g)
    H_e = H_s @ (torch.randn(192, 192, generator=g) * 0.1) + torch.randn(400, 39, 192, generator=g) * 0.05
    W, b, info = fit_per_feature_auto(H_s, H_e, device="cpu")
    assert torch.isfinite(W).all() and torch.isfinite(b).all()
    worst = 0.0
    for f, lam in enumerate(info["ridge_lambda_per_feature"]):
        W_f = fit_ridge(H_s[:, f, :], H_e[:, f, :], lam)[0]
        worst = max(worst, float((W_f - W[f]).norm() / W_f.norm()))
    print(worst)
""")


def test_cpu_per_feature_ridge_survives_set_num_threads():
    proc = subprocess.run([sys.executable, "-c", _SUBPROCESS], capture_output=True, text=True,
                          timeout=300, env=dict(os.environ))
    assert proc.returncode == 0, f"CPU per-feature ridge fit crashed:\n{proc.stderr[-2000:]}"
    assert "SLASWP" not in proc.stderr, "MKL batched-LU error printed on the CPU path"
    worst_rel = float(proc.stdout.strip().splitlines()[-1])
    assert worst_rel < 1e-3, f"CPU batched fit disagrees with serial fit_ridge: rel {worst_rel:.2e}"


def test_cpu_and_cuda_paths_agree():
    if not torch.cuda.is_available():
        return  # CPU-only machine: the subprocess test above covers the CPU path itself
    from shallow_tfm.translators import fit_per_feature_auto
    g = torch.Generator().manual_seed(1)
    H_s = torch.randn(300, 12, 64, generator=g)
    H_e = H_s @ (torch.randn(64, 64, generator=g) * 0.1) + torch.randn(300, 12, 64, generator=g) * 0.05
    W_c, b_c, i_c = fit_per_feature_auto(H_s, H_e, device="cpu")
    W_g, b_g, i_g = fit_per_feature_auto(H_s, H_e, device="cuda")
    assert i_c["ridge_lambda_per_feature"] == i_g["ridge_lambda_per_feature"]
    assert float((W_c - W_g).norm() / W_c.norm()) < 1e-3
