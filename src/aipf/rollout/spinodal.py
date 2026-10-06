"""Spinodal decomposition from a cube: integrate the trained model from frame 0 of a measured run and
compare ``Phi(t)``, ``S_cc(k, t)`` and ``L(t)`` with that run. Both sides go through one reconstruction
(scatter the modes, sigma filter, declared grid). Noisy: one rollout per seed, compared on seed-invariant
metrics. Also holds what the slab driver shares: declarations, the model, the archive, the output tier."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

from aipf.paths import PUBLISHED_DIRNAME
from aipf.solve.noise import declared_noise
from aipf.solve.trust_domain import TrustDomain
from aipf.spectral import SpectralOps
from aipf.system import System
from aipf.train.dataset import scatter_modes

from .imex import rollout_imex
from .observables import decomp_metrics

#: Solver knobs every driver reads from the declaration.
SOLVER_KEYS = ("clamp_rho", "state_proj", "state_clamp", "mass_restore",
               "noise_eval", "noise_scale", "predictor_floor")
#: Archive keys, as ``aipf.train.dataset.ArchiveKeys`` names them.
ARCHIVE_KEYS = ("file_name", "amplitudes", "amplitudes_channel_axis", "labels",
                "box", "temperature", "frame_interval", "quality_file",
                "quality_key")
#: Per-driver keys.
DRIVER_KEYS = {"spinodal": ("modes_tree", "grid", "field_stride"),
               "slab": ("modes_tree", "grid", "field_stride", "profile_axis")}


def declared(system: System, driver: str,
             override: Optional[Mapping[str, Any]] = None) -> dict:
    """The declaration one driver runs with: ``system.defaults["rollout"]`` (``"solver"``, ``"archive"``,
    ``<driver>``) or ``override`` of the same shape; every key required, none defaulted."""
    block = override if override is not None else \
        system.defaults.get("rollout")
    if block is None:
        raise ValueError(
            f"system {system.name!r} declares no defaults['rollout']: the "
            f"solver knobs {list(SOLVER_KEYS)}, the archive keys "
            f"{list(ARCHIVE_KEYS)} and the {driver!r} keys "
            f"{list(DRIVER_KEYS[driver])}.")
    out = {}
    for part, keys in (("solver", SOLVER_KEYS), ("archive", ARCHIVE_KEYS),
                       (driver, DRIVER_KEYS[driver])):
        given = dict(block.get(part, {}))
        missing = [k for k in keys if k not in given]
        extra = sorted(set(given) - set(keys))
        if missing or extra:
            raise ValueError(
                f"rollout declaration [{part!r}] of system {system.name!r}: "
                f"missing {missing}, not read {extra}")
        out[part] = given
    return out


def trust_domain(system: System) -> TrustDomain:
    """The density trapezoid of ``system.trust_domain``; the rollout projects at fixed ``T``, so
    ``T_range`` is not carried. A system that declares no domain is refused."""
    domain = system.trust_domain
    if domain is None:
        raise ValueError(
            f"system {system.name!r} declares no trust_domain: the rollout "
            f"state projection needs the density trapezoid")
    return TrustDomain(inner=domain.inner, outer=domain.outer)


def load_model(system: System, path: Path) -> torch.nn.Module:
    """The declared functional with this file's weights, float32, eval mode."""
    from aipf.functional.build import build
    from aipf.train.checkpoint_formats import load_lightning_hparams_into

    saved = torch.load(path, map_location="cpu", weights_only=False)
    state = saved["state_dict"] if "state_dict" in saved else \
        saved["model_state_dict"]
    model = build(system)
    if set(state) <= set(model.state_dict()):
        merged = dict(model.state_dict())
        merged.update(state)
        model.load_state_dict(merged, strict=True)
    else:
        load_lightning_hparams_into(model, state)
    return model.eval()


def resolve_checkpoint(system: System, ckpt) -> Path:
    """``"published"`` (``PUBLISHED_DIRNAME``): the system's checkpoint, md5-proved; otherwise a path."""
    if ckpt == PUBLISHED_DIRNAME:
        return system.resolve_checkpoint()
    path = Path(ckpt)
    if not path.is_file():
        raise FileNotFoundError(f"no checkpoint at {path}")
    return path


@dataclass(frozen=True)
class Measured:
    """One archived run on the declared grid: ``rho_hat (F, n, Gx, Gy, Gzr)`` sigma-filtered, its real
    fields, per-frame boxes, temperature and frame interval."""

    rho_hat: torch.Tensor
    rho: np.ndarray
    boxes: np.ndarray
    T: float
    frame_interval: float


def read_run(system: System, decl: dict, driver: str, run: str) -> Measured:
    """Scatter one archived run onto ``decl[driver]["grid"]`` and apply the sigma filter to every frame.

    A run whose archive records a reference cell (:data:`aipf.pipeline.modes.REFERENCE_BOX_KEY`) is read
    in it: every frame's ``V``, ``k`` and returned box is the cell's."""
    from aipf.pipeline.modes import REFERENCE_BOX_KEY

    arch = decl["archive"]
    grid = tuple(int(g) for g in decl[driver]["grid"])
    run_dir = system.paths.raw() / decl[driver]["modes_tree"] / run
    with np.load(run_dir / arch["file_name"]) as z:
        n_frames = len(z[arch["box"]])
        hi = n_frames
        qc = (run_dir / arch["quality_file"]) if arch["quality_file"] else None
        if qc is not None and qc.is_file():
            hi = int(json.loads(qc.read_text())[arch["quality_key"]])
        boxes = np.asarray(z[arch["box"]][:hi])
        if REFERENCE_BOX_KEY in z:
            cell = np.asarray(z[REFERENCE_BOX_KEY], dtype=np.float64)
            boxes = np.repeat(cell[None, :], len(boxes), axis=0)
        amps = np.asarray(z[arch["amplitudes"]][:hi])
        if int(arch["amplitudes_channel_axis"]) == 2:
            amps = np.moveaxis(amps, -1, -2)
        rho_hat = scatter_modes(amps, z[arch["labels"]],
                                boxes.prod(axis=1)[:, None, None], grid)
        T = float(z[arch["temperature"]])
        frame_interval = float(z[arch["frame_interval"]])
    h = torch.tensor(rho_hat)
    ops = SpectralOps(grid, system.n_species,
                      nyquist_mask=system.functional.kwargs["nyquist_mask"])
    filt = ops.sigma_filter(torch.tensor(boxes, dtype=torch.float32),
                            float(system.defaults["sigma"]))
    h = h * filt.to(h.dtype)
    N = grid[0] * grid[1] * grid[2]
    return Measured(h, ops.irfft(h * N).numpy(), boxes, T, frame_interval)


def noise_for(system: System, decl: dict, T: float, eps: float = 1.0) -> dict:
    """The noise dict :func:`rollout_imex` takes: FDT amplitude ``eps kB T``, the declared convention."""
    nd = declared_noise(system)
    s = decl["solver"]
    return {"kBT_noise": eps * float(system.constants["kB"]) * float(T),
            "noise_scale": float(s["noise_scale"]),
            "noise_mode": nd["noise_mode"], "sigma_noise": nd["sigma_noise"],
            "noise_eval": s["noise_eval"],
            "predictor_floor": float(s["predictor_floor"])}


def run_rollout(system: System, model, decl: dict, h0: torch.Tensor,
                box0: np.ndarray, T: float, *, t_end: float, dt: float,
                save_ps: float, seed: Optional[int], device: str,
                eps: float = 1.0, state_proj: Optional[str] = None,
                state_clamp=None) -> torch.Tensor:
    """One rollout under the declaration; ``seed=None`` is deterministic. Returns CPU states."""
    s = decl["solver"]
    n_steps = int(round(t_end / dt))
    save_every = max(1, int(round(save_ps / dt)))
    generator = None
    noise = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        noise = noise_for(system, decl, T, eps)
    return rollout_imex(
        model, h0.to(device), torch.tensor(box0, dtype=torch.float32), T, dt,
        n_steps, kB=float(system.constants["kB"]),
        m_stab=declared_noise(system)["m_stab"],
        state_proj=state_proj or s["state_proj"],
        state_clamp=s["state_clamp"] if state_clamp is None else state_clamp,
        domain=trust_domain(system), mass_restore=s["mass_restore"],
        clamp_rho=s["clamp_rho"], noise=noise, save_every=save_every,
        generator=generator)


def output_dir(system: System, driver: str, ckpt_md5: str,
               record: Mapping[str, Any], out: Optional[Path]) -> Path:
    """``<out or data/<sys>/rollout>/<md5 of the record>[:12]``."""
    key = json.dumps({"driver": driver, "ckpt": ckpt_md5, **record},
                     sort_keys=True, default=repr)
    root = Path(out) if out is not None else \
        system.paths.data_root() / "rollout"
    d = root / hashlib.md5(key.encode()).hexdigest()[:12]
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_manifest(out_dir: Path, system: System, driver: str, path: Path,
                   ckpt_md5: str, record: Mapping[str, Any],
                   outputs: Sequence[str]) -> None:
    (out_dir / "MANIFEST.json").write_text(json.dumps({
        "system": system.name, "driver": driver, "checkpoint": str(path),
        "md5": ckpt_md5, "noise": declared_noise(system), "record": record,
        "outputs": list(outputs),
        "written_at": datetime.now(timezone.utc).isoformat()},
        indent=1, default=repr))


def _file_md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def spinodal(system: System, ckpt, *, run: str, seeds: Sequence[int],
             t_end: float, dt: float, save_ps: float, device: str,
             out: Optional[Path] = None,
             declaration: Optional[Mapping[str, Any]] = None) -> Path:
    """Roll the cube ``run`` once per seed (``seeds`` empty: one deterministic rollout) and write
    ``spinodal_<run>_<tag>.npz`` plus ``MANIFEST.json``; returns the output directory."""
    decl = declared(system, "spinodal", declaration)
    path = resolve_checkpoint(system, ckpt)
    md5 = _file_md5(path)
    record = {"run": run, "seeds": list(seeds), "t_end": t_end, "dt": dt,
              "save_ps": save_ps, "declaration": decl}
    out_dir = output_dir(system, "spinodal", md5, record, out)
    model = load_model(system, path).to(device)
    md = read_run(system, decl, "spinodal", run)
    xc = int(system.table_keys["x_channel"])
    t_md = np.arange(len(md.rho)) * md.frame_interval
    Phi_md, L_md, Sk_md = decomp_metrics(md.rho, md.boxes.mean(axis=1), xc)
    N = int(np.prod(decl["spinodal"]["grid"]))
    ops = SpectralOps(tuple(decl["spinodal"]["grid"]), system.n_species,
                      nyquist_mask=model.ops.nyquist_mask)
    written = []
    for seed in (list(seeds) or [None]):
        traj = run_rollout(system, model, decl, md.rho_hat[0:1], md.boxes[0],
                           md.T, t_end=t_end, dt=dt, save_ps=save_ps,
                           seed=seed, device=device)
        rho_m = ops.irfft(traj * N).numpy()
        t_m = np.arange(len(rho_m)) * save_ps
        Phi_m, L_m, Sk_m = decomp_metrics(
            rho_m, np.full(len(rho_m), md.boxes[0].mean()), xc)
        fs = int(decl["spinodal"]["field_stride"])
        n_common = min(len(rho_m), len(md.rho))

        def c_field(rho):
            return (rho[:, xc] / np.maximum(rho.sum(axis=1), 1e-9)
                    ).astype(np.float32)

        tag = "det" if seed is None else f"s{seed}"
        name = f"spinodal_{run}_{tag}.npz"
        np.savez(out_dir / name, t_model=t_m, Phi_model=Phi_m, L_model=L_m,
                 Sk_model=Sk_m, t_md=t_md, Phi_md=Phi_md, L_md=L_md,
                 Sk_md=Sk_md, box_L=md.boxes[0].mean(), T_K=md.T,
                 t_fields=t_m[:n_common:fs],
                 c_model=c_field(rho_m[:n_common:fs]),
                 c_md=c_field(md.rho[:n_common:fs]),
                 rho_hat_final=traj[-1].numpy())
        written.append(name)
    write_manifest(out_dir, system, "spinodal", path, md5, record, written)
    return out_dir
