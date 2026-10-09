"""``--precision`` on ``aipf rollout`` and the drivers: fp32 by default, exactly as before; fp64 on request.

The default leaves every existing invocation as it was: the driver's record (what its output directory is
hashed from and its manifest holds) carries no precision key, so the directory and the manifest are the
ones a run always wrote. ``fp64`` casts the model and the initial state before the solver, which then runs
float64 end to end, and is recorded, so its output directory differs. The model is the factory toy of
``tests/toy_factory_model.py`` on a ``(4, 4, 4)`` grid; the archive read is replaced by a field in memory."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math

import numpy as np
import pytest
import torch

import toy_factory_model as toy
from aipf.functional.build import build
from aipf.rollout import fdt as fdt_mod
from aipf.rollout import spinodal as spinodal_mod
from aipf.rollout.spinodal import Measured, output_dir, run_record, run_rollout
from aipf.solve.precision import PRECISIONS, cast_pair, check_precision
from aipf.system import Noise, TrustDomain

GRID = toy.GRID
DECL = {
    "solver": dict(clamp_rho=1e-3, state_proj="floor", state_clamp=1e-3, mass_restore="shift",
                   noise_eval="ito", noise_scale=1.0, predictor_floor=1e-4),
    "archive": dict(file_name="m.npz", amplitudes="a", amplitudes_channel_axis=2, labels="n",
                    box="b", temperature="T", frame_interval="dt", quality_file=None,
                    quality_key=None),
    "spinodal": dict(modes_tree="x", grid=GRID, field_stride=1),
}
#: The record an fp32 spinodal run of these arguments writes, spelt out, and the directory it hashes to
#: under checkpoint md5 ``"0" * 32``; ``output_dir`` is unchanged, so the digest is the one it always was.
OLD_RECORD = {"run": "r", "seeds": [3], "t_end": 0.05, "dt": 0.01, "save_ps": 0.02,
              "declaration": DECL}
OLD_DIGEST = hashlib.md5(json.dumps({"driver": "spinodal", "ckpt": "0" * 32, **OLD_RECORD},
                                    sort_keys=True, default=repr).encode()).hexdigest()[:12]


@pytest.fixture(autouse=True)
def _one_thread():
    """Fields of a few dozen cells: a thread pool only adds its overhead, and on a shared machine a lot of it."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


@pytest.fixture
def demo(tmp_path):
    """The two-channel factory demo, with the noise and trust domain the drivers read."""
    base = toy.demo_system(str(tmp_path / "raw"))
    return dataclasses.replace(base, noise=Noise(mode="none", m_stab="mean"),
                               trust_domain=TrustDomain(inner=(0.2, 0.2), outer=(2.0, 2.0)),
                               variants={})


def _field(seed=3):
    g = torch.Generator().manual_seed(seed)
    rho = torch.stack([0.50 + 0.02 * torch.randn(GRID, generator=g),
                       0.40 + 0.02 * torch.randn(GRID, generator=g)]).unsqueeze(0)
    return torch.fft.rfftn(rho, dim=(-3, -2, -1)) / math.prod(GRID)


def _model(demo):
    torch.manual_seed(0)
    return build(demo).eval()


# ---------------------------------------------------------------------------
# the CLI flag
# ---------------------------------------------------------------------------

_ARGV = ["rollout", "slab", "--system", "lj", "--ckpt", "published", "--run", "r", "--seeds",
         "--t-end", "1", "--dt", "1", "--save-ps", "1", "--device", "cpu", "--out", "data"]


def test_the_rollout_flag_defaults_to_fp32_and_takes_fp64_and_nothing_else():
    from aipf.cli.main import build_parser
    assert build_parser().parse_args(_ARGV).precision == "fp32"
    assert build_parser().parse_args(_ARGV + ["--precision", "fp64"]).precision == "fp64"
    with pytest.raises(SystemExit):
        build_parser().parse_args(_ARGV + ["--precision", "fp16"])


@pytest.mark.parametrize("flag, want", [([], "fp32"), (["--precision", "fp64"], "fp64")])
def test_the_rollout_verb_hands_the_driver_its_precision(monkeypatch, tmp_path, flag, want):
    import aipf.cli.rollout_cmd as rollout_cmd
    import aipf.rollout.slab as slab_mod
    from aipf.cli.main import main
    seen = {}
    monkeypatch.setattr(rollout_cmd, "refusal", lambda system, driver, ckpt: None)
    monkeypatch.setattr(slab_mod, "slab", lambda system, ckpt, **kw: seen.update(kw) or tmp_path)
    assert main(_ARGV + flag) == 0
    assert seen["precision"] == want


# ---------------------------------------------------------------------------
# the record and the output directory
# ---------------------------------------------------------------------------

def test_an_fp32_record_and_its_directory_are_the_ones_a_run_always_wrote(tmp_path):
    record = run_record("r", (3,), 0.05, 0.01, 0.02, DECL)
    assert record == OLD_RECORD and list(record) == list(OLD_RECORD)
    assert run_record("r", (3,), 0.05, 0.01, 0.02, DECL, "fp32") == OLD_RECORD
    assert output_dir(None, "spinodal", "0" * 32, record, tmp_path).name == OLD_DIGEST


def test_an_fp64_record_names_its_precision_and_gets_its_own_directory(tmp_path):
    record = run_record("r", (3,), 0.05, 0.01, 0.02, DECL, "fp64")
    assert record == {**OLD_RECORD, "precision": "fp64"}
    assert output_dir(None, "spinodal", "0" * 32, record, tmp_path).name != OLD_DIGEST
    with pytest.raises(ValueError, match="precision must be one of \\['fp32', 'fp64'\\]"):
        run_record("r", (3,), 0.05, 0.01, 0.02, DECL, "double")


def test_cast_pair_leaves_fp32_untouched_and_casts_fp64():
    model = torch.nn.Linear(2, 2)
    h = _field()
    m, s = cast_pair(model, h, "fp32")
    assert m is model and s is h and model.weight.dtype == torch.float32
    m, s = cast_pair(model, h, "fp64")
    assert m is not model and m.weight.dtype == torch.float64 and s.dtype == torch.complex128
    assert model.weight.dtype == torch.float32                   # the caller's model is a float32 one still
    assert torch.equal(m.weight, model.weight.double())
    assert torch.equal(s, h.to(torch.complex128))
    assert PRECISIONS[check_precision("fp64")] == (torch.float64, torch.complex128)


# ---------------------------------------------------------------------------
# the driver API reaches the solver in the declared precision
# ---------------------------------------------------------------------------

def _spy_solver(monkeypatch):
    seen = []
    real = spinodal_mod.rollout_imex

    def spy(model, rho_hat0, box, *args, **kwargs):
        seen.append({"params": {p.dtype for p in model.parameters()}, "ops": model.ops.NX.dtype,
                     "state": rho_hat0.dtype, "box": box.dtype})
        return real(model, rho_hat0, box, *args, **kwargs)

    monkeypatch.setattr(spinodal_mod, "rollout_imex", spy)
    return seen


@pytest.mark.parametrize("seed", [None, 5])
def test_run_rollout_in_fp64_hands_the_solver_a_float64_pair(demo, monkeypatch, seed):
    seen = _spy_solver(monkeypatch)
    model = _model(demo)
    h0 = _field()
    traj = run_rollout(demo, model, DECL, h0, np.array([8.0, 8.0, 8.0]), 1.0, t_end=0.05,
                       dt=0.01, save_ps=0.02, seed=seed, device="cpu", precision="fp64")
    assert seen == [{"params": {torch.float64}, "ops": torch.float64, "state": torch.complex128,
                     "box": torch.float64}]
    assert traj.dtype == torch.complex128 and traj.shape == (4, 2, 4, 4, 3)
    assert torch.equal(traj[:, :, 0, 0, 0].real,
                       h0[0, :, 0, 0, 0].real.double().expand(4, 2))


def test_an_fp64_run_leaves_the_callers_model_for_a_later_fp32_one(demo):
    model = _model(demo)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    h0, box = _field(), np.array([8.0, 8.0, 8.0])
    kw = dict(t_end=0.05, dt=0.01, save_ps=0.02, seed=None, device="cpu")
    fresh = run_rollout(demo, _model(demo), DECL, h0, box, 1.0, **kw)
    run_rollout(demo, model, DECL, h0, box, 1.0, precision="fp64", **kw)
    after = model.state_dict()
    assert all(after[k].dtype == before[k].dtype and torch.equal(after[k], before[k]) for k in before)
    assert torch.equal(run_rollout(demo, model, DECL, h0, box, 1.0, **kw), fresh)


def test_run_rollout_by_default_hands_the_solver_what_it_always_did(demo, monkeypatch):
    seen = _spy_solver(monkeypatch)
    model = _model(demo)
    h0 = _field()
    a = run_rollout(demo, model, DECL, h0, np.array([8.0, 8.0, 8.0]), 1.0, t_end=0.05, dt=0.01,
                    save_ps=0.02, seed=None, device="cpu")
    b = spinodal_mod.rollout_imex(
        model, h0, torch.tensor(np.array([8.0, 8.0, 8.0]), dtype=torch.float32), 1.0, 0.01, 5,
        kB=1.0, m_stab="mean", state_proj="floor", state_clamp=1e-3,
        domain=spinodal_mod.trust_domain(demo), mass_restore="shift", clamp_rho=1e-3,
        noise=None, save_every=2, generator=None)
    assert seen[0] == {"params": {torch.float32}, "ops": torch.float32, "state": torch.complex64,
                       "box": torch.float32}
    assert torch.equal(a, b)


def test_the_spinodal_driver_writes_fp64_to_its_own_recorded_directory(demo, monkeypatch, tmp_path):
    """End to end through ``spinodal()``, the checkpoint read from a file; only the archive is in memory."""
    model = _model(demo)
    ckpt = tmp_path / "toy.ckpt"
    torch.save({"state_dict": model.state_dict()}, ckpt)
    frames = torch.cat([_field(seed) for seed in (3, 4, 5)])
    rho = torch.fft.irfftn(frames * math.prod(GRID), s=GRID, dim=(-3, -2, -1)).numpy()
    md = Measured(frames, rho, np.full((3, 3), 8.0), 1.0, 0.02)
    monkeypatch.setattr(spinodal_mod, "read_run", lambda system, decl, driver, run: md)
    seen = _spy_solver(monkeypatch)
    out = {}
    for precision in ("fp32", "fp64"):
        out[precision] = spinodal_mod.spinodal(
            demo, str(ckpt), run="r", seeds=(), t_end=0.04, dt=0.01, save_ps=0.02,
            device="cpu", out=tmp_path / "out", declaration=DECL, precision=precision)
    assert out["fp32"] != out["fp64"]
    assert [s["state"] for s in seen] == [torch.complex64, torch.complex128]
    assert [s["params"] for s in seen] == [{torch.float32}, {torch.float64}]
    m32 = json.loads((out["fp32"] / "MANIFEST.json").read_text())
    m64 = json.loads((out["fp64"] / "MANIFEST.json").read_text())
    assert "precision" not in m32["record"] and m64["record"]["precision"] == "fp64"
    assert {k: v for k, v in m64["record"].items() if k != "precision"} == m32["record"]
    final32 = np.load(out["fp32"] / "spinodal_r_det.npz")["rho_hat_final"]
    final64 = np.load(out["fp64"] / "spinodal_r_det.npz")["rho_hat_final"]
    assert final32.dtype == np.complex64 and final64.dtype == np.complex128
    assert np.abs(final64 - final32).max() < 1e-5


def test_the_fdt_gate_in_fp64_builds_its_state_and_hessian_in_float64(demo, monkeypatch):
    rho_bar = np.array([0.5, 0.4])
    h = fdt_mod.homogeneous_state(rho_bar, GRID, "cpu", torch.complex128)
    assert h.dtype == torch.complex128 and torch.equal(h[0, :, 0, 0, 0].real,
                                                      torch.tensor(rho_bar))
    assert fdt_mod.homogeneous_state(rho_bar, GRID, "cpu").dtype == torch.complex64
    seen = []
    real = fdt_mod.rollout_imex
    monkeypatch.setattr(fdt_mod, "rollout_imex",
                        lambda model, h0, box, *a, **kw: seen.append((h0.dtype, box.dtype))
                        or real(model, h0, box, *a, **kw))
    demo = dataclasses.replace(demo, noise=Noise(mode="gaussian", m_stab="mean"))
    model = _model(demo)
    res = fdt_mod.run_one(demo, model, DECL, box=np.array([8.0, 8.0, 8.0]), rho_bar=rho_bar,
                          grid=GRID, T=1.0, eps=0.01, noise_eval="ito", t_end=0.05, dt=0.01,
                          save_dt=0.01, burn_in=0.2, k_band=2.0, seed=0, device="cpu",
                          precision="fp64")
    assert seen == [(torch.complex128, torch.float64)]
    assert next(model.parameters()).dtype == torch.float32      # run on a copy
    assert np.isfinite(res["r_tr_mode"]).all()
