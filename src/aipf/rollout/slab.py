"""Coexistence from a slab: integrate the trained model from frame 0 of a phase-separated run and compare
the profile ``c(z)``, its plateaus, the 10-90 interface width and ``Phi(t)`` with that run. ``seeds``
empty: deterministic. The profile axis is declared (``profile_axis``)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from aipf.spectral import SpectralOps
from aipf.system import System

from .observables import (conc_profile, extract_plateaus, field_frames,
                          interface_width, weighted_var_c)
from .spinodal import (_file_md5, declared, load_model, output_dir, read_run,
                       resolve_checkpoint, run_record, run_rollout,
                       write_manifest)


def slab(system: System, ckpt, *, run: str, seeds: Sequence[int],
         t_end: float, dt: float, save_ps: float, device: str,
         out: Optional[Path] = None,
         declaration: Optional[Mapping[str, Any]] = None,
         precision: str = "fp32") -> Path:
    """Roll the slab ``run`` (once per seed, or once deterministically) and write ``<run>_<tag>.npz`` plus
    ``MANIFEST.json``; returns the output directory. ``precision`` as
    :func:`aipf.rollout.spinodal.spinodal`."""
    decl = declared(system, "slab", declaration)
    path = resolve_checkpoint(system, ckpt)
    md5 = _file_md5(path)
    record = run_record(run, seeds, t_end, dt, save_ps, decl, precision)
    out_dir = output_dir(system, "slab", md5, record, out)
    model = load_model(system, path).to(device)
    md = read_run(system, decl, "slab", run)
    grid = tuple(int(g) for g in decl["slab"]["grid"])
    axis = int(decl["slab"]["profile_axis"])
    xc = int(system.table_keys["x_channel"])
    ops = SpectralOps(grid, system.n_species, nyquist_mask=model.ops.nyquist_mask)
    N = grid[0] * grid[1] * grid[2]
    t_md = np.arange(len(md.rho)) * md.frame_interval
    Phi_md, c_all, _ = weighted_var_c(md.rho, xc)
    prof_md = conc_profile(md.rho, xc, axis)
    x_poor_md, x_rich_md = extract_plateaus(prof_md[-1])
    w_md = interface_width(prof_md[-1], float(md.boxes[-1][axis]) / grid[axis])
    written = []
    for seed in (list(seeds) or [None]):
        traj = run_rollout(system, model, decl, md.rho_hat[0:1], md.boxes[0],
                           md.T, t_end=t_end, dt=dt, save_ps=save_ps,
                           seed=seed, device=device, precision=precision)
        rho_m = ops.irfft(traj * N).numpy()
        t_m = np.arange(len(rho_m)) * save_ps
        sel_m, idx_md = field_frames(t_m, t_md, decl["slab"]["field_stride"])
        Phi_m, c_m, _ = weighted_var_c(rho_m, xc)
        prof_m = conc_profile(rho_m, xc, axis)
        x_poor_m, x_rich_m = extract_plateaus(prof_m[-1])
        w_m = interface_width(prof_m[-1], float(md.boxes[0][axis]) / grid[axis])
        tag = "det" if seed is None else f"s{seed}"
        name = f"{run}_{tag}.npz"
        np.savez(out_dir / name, t_model=t_m, t_md=t_md, prof_model=prof_m,
                 prof_md=prof_md, x_poor_model=x_poor_m, x_rich_model=x_rich_m,
                 x_poor_md=x_poor_md, x_rich_md=x_rich_md, width_model=w_m,
                 width_md=w_md, Phi_model=Phi_m, Phi_md=Phi_md,
                 box_L=np.asarray(md.boxes[0], dtype=np.float64), T_K=md.T,
                 t_fields=t_m[sel_m], c_model=c_m[sel_m].astype(np.float32),
                 c_md=c_all[idx_md].astype(np.float32),
                 rho_hat_final=traj[-1].numpy())
        written.append(name)
    write_manifest(out_dir, system, "slab", path, md5, record, written)
    return out_dir
