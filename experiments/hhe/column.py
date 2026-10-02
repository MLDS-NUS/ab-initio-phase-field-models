"""The hydrogen-helium rain column: the learned dynamics in a periodic box under mass-weighted gravity,
either a walled column or the wall-free mirror cuboid, at one temperature or on a profile T(z), from a
seeded uniform state or from the last field of a previous chunk.

The scheme is ``aipf.rollout.imex`` with its ``v_ext`` and ``kbt_field``; the potentials, the profiles,
the diagnostics and the chunking are this system's. Every knob is declared in ``defaults["column"]`` or is
a required argument."""
from __future__ import annotations

import argparse
import json
import warnings
import zipfile
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

from aipf.rollout.imex import rollout_imex
from aipf.rollout.spinodal import (_file_md5, load_model, noise_for, output_dir,
                                   resolve_checkpoint, trust_domain,
                                   write_manifest)
from aipf.solve.noise import check_m_stab, declared_noise
from aipf.solve.trust_domain import in_domain
from aipf.spectral import SpectralOps
from aipf.system import Noise, System

GEOMETRIES = ("walled", "mirror")
PROFILES = ("linear", "step")
SOLVER_KEYS = ("clamp_rho", "state_proj", "state_clamp", "mass_restore",
               "noise_eval", "noise_scale", "predictor_floor")
COLUMN_KEYS = ("noise", "masses", "T_window", "T_margin", "ramp_min_cells",
               "quarantine_thresh", "seed_chunk_stride", "rain_rule", "solver")
#: Knobs that define the trajectory a chunk continues; a source differing in any is refused.
COMPAT_KEYS = ("geometry", "grid", "dx", "bond", "wall_amp", "wall_width", "T",
               "x_he", "rho_tot", "t_profile", "noisy", "dt", "ckpt_md5",
               "declaration", "m_stab")
NPZ_KEYS = (
    "t", "prof_H", "prof_He", "Phi", "z_he_com", "mass_H", "mass_He",
    "occ_interior", "occ_wall", "rho_min", "rain_band", "rain_rule",
    "t_profile", "t_fields", "rho_fields", "interior_mask", "uniform_g_mask",
    "z", "V_wall", "G", "g_eff", "bond", "L_free", "wall_width",
    "wall_amp_kbt", "kBT", "dz", "T_K", "sigma", "box", "grid", "params",
)


def declared_column(system: System,
                    override: Optional[Mapping[str, Any]] = None) -> dict:
    """``system.defaults["column"]`` (or ``override``), every key required, none read twice."""
    block = override if override is not None else system.defaults.get("column")
    if block is None:
        raise ValueError(f"system {system.name!r} declares no defaults['column']")
    for keys, given, part in ((COLUMN_KEYS, block, "column"),
                              (SOLVER_KEYS, block.get("solver", {}), "solver")):
        missing = [k for k in keys if k not in given]
        extra = sorted(set(given) - set(keys))
        if missing or extra:
            raise ValueError(f"column declaration [{part!r}]: missing {missing}, "
                             f"not read {extra}")
    if not isinstance(block["noise"], Noise):
        raise ValueError("column declaration ['noise'] must be a Noise(mode, m_stab)")
    return {k: (dict(v) if isinstance(v, Mapping) else v) for k, v in block.items()}


# -- geometry: the periodic C1 potentials and the T profiles --

def _z_axis(grid, box):
    grid = tuple(int(g) for g in grid)
    box = tuple(float(b) for b in box)
    if len(grid) != 3 or len(box) != 3 or min(grid) < 1 or min(box) <= 0.0:
        raise ValueError(f"grid and box must be three positive values, got {grid}, {box}")
    if grid[2] % 2:
        raise ValueError(f"Gz must be even (rfft grid), got {grid[2]}")
    Gz, Lz = grid[2], box[2]
    dz = Lz / Gz
    z = np.arange(Gz, dtype=np.float64) * dz
    u = (z + 0.5 * Lz) % Lz - 0.5 * Lz
    return grid, box, z, u, dz


def _v_ext(grid, masses, g_eff, G, V_wall):
    m_H, m_He = masses
    V_H = m_H * g_eff * G + V_wall
    V_He = m_He * g_eff * G + V_wall
    prof = np.stack([V_H, V_He]).astype(np.float32)
    return torch.from_numpy(prof)[:, None, None, :].expand(2, *grid).contiguous()


def walled_vext(grid, box, T, *, bond, wall_amp_kbt, wall_width, quarantine_thresh,
                kB, masses):
    """``V_i = m_i g_eff G(z) + V_wall(z)``: raised-cosine wall straddling the wrap, gravity uniform over
    the free column ``Lz - 2w``, ``Bo = (m_He - m_H) g_eff L_free / kBT``. Returns ``(v_ext, interior, info)``."""
    grid, box, z, u, dz = _z_axis(grid, box)
    Lz, w = box[2], float(wall_width)
    if not 0.0 < w < 0.5 * Lz:
        raise ValueError(f"wall_width must satisfy 0 < w < Lz/2 = {0.5 * Lz}, got {w}")
    if float(wall_amp_kbt) < 0.0 or float(quarantine_thresh) < 0.0:
        raise ValueError("wall_amp_kbt and quarantine_thresh must be >= 0")
    kBT = kB * float(T)
    in_wall = np.abs(u) <= w
    V_wall = np.where(in_wall, float(wall_amp_kbt) * kBT * 0.5
                      * (1.0 + np.cos(np.pi * u / w)), 0.0)
    if float(wall_amp_kbt) == 0.0:
        V_wall = np.zeros(grid[2], dtype=np.float64)
    G = np.where(in_wall, u * (1.0 - Lz / (2.0 * w))
                 - (Lz / (2.0 * np.pi)) * np.sin(np.pi * u / w),
                 u - np.sign(u) * 0.5 * Lz)
    G = G - G.mean()
    L_free = Lz - 2.0 * w
    g_eff = float(bond) * kBT / ((masses[1] - masses[0]) * L_free)
    interior = np.ascontiguousarray(V_wall <= float(quarantine_thresh) * kBT)
    info = dict(g_eff=g_eff, bond=float(bond), L_free=L_free, wall_width=w,
                wall_amp_kbt=float(wall_amp_kbt), kBT=kBT, dz=dz, z=z,
                V_wall=V_wall, G=G,
                uniform_g_mask=np.ascontiguousarray(np.abs(u) >= w))
    return _v_ext(grid, masses, g_eff, G, V_wall), interior, info


def _mirror_G(u, Lz, w):
    a = np.abs(u)
    cap = (2.0 * w / np.pi) * (np.cos(np.pi * a / (2.0 * w)) - 1.0)
    lin = -2.0 * w / np.pi - (a - w)
    base = -2.0 * w / np.pi - (0.5 * Lz - 2.0 * w)
    bot = base - (2.0 * w / np.pi) * np.cos(np.pi * (0.5 * Lz - a) / (2.0 * w))
    return np.where(a <= w, cap, np.where(a <= 0.5 * Lz - w, lin, bot))


def mirror_vext(grid, box, T, *, bond, smooth_width, kB, masses):
    """Wall-free even triangle wave: potential maximum at the wrap (top), minimum at ``z = Lz/2``
    (centre), force smoothed over ``smooth_width``; ``Bo`` on the half column ``Lz/2 - 2w``."""
    grid, box, z, u, dz = _z_axis(grid, box)
    Lz, w = box[2], float(smooth_width)
    if not 0.0 < w < 0.25 * Lz:
        raise ValueError(f"smooth_width must satisfy 0 < w < Lz/4 = {0.25 * Lz}, got {w}")
    kBT = kB * float(T)
    G = _mirror_G(u, Lz, w)
    G = G - G.mean()
    L_free = 0.5 * Lz - 2.0 * w
    g_eff = float(bond) * kBT / ((masses[1] - masses[0]) * L_free)
    V_wall = np.zeros(grid[2], dtype=np.float64)
    a = np.abs(u)
    info = dict(g_eff=g_eff, bond=float(bond), L_free=L_free, wall_width=w,
                wall_amp_kbt=0.0, kBT=kBT, dz=dz, z=z, V_wall=V_wall, G=G,
                uniform_g_mask=np.ascontiguousarray((a >= w) & (a <= 0.5 * Lz - w)))
    m_H, m_He = masses
    prof = np.stack([m_H * g_eff * G, m_He * g_eff * G]).astype(np.float32)
    v = torch.from_numpy(prof)[:, None, None, :].expand(2, *grid).contiguous()
    return v, np.ones(grid[2], dtype=bool), info


def _check_t_window(name, t_prof, t_range):
    if t_range is None:
        return
    lo, hi = float(t_range[0]), float(t_range[1])
    if not (lo <= float(t_prof.min()) and float(t_prof.max()) <= hi):
        raise ValueError(f"{name}: the sampled profile spans [{t_prof.min():.1f}, "
                         f"{t_prof.max():.1f}] K, outside the window [{lo:.1f}, {hi:.1f}] K "
                         f"(kbt_field is full-grid, blend cells included)")


def _kbt(grid, t_prof, kB):
    full = np.broadcast_to(t_prof, (grid[0], grid[1], grid[2]))
    return torch.from_numpy(np.ascontiguousarray((kB * full).astype(np.float32)))


def t_profile_walled(grid, box, t_top, t_bottom, *, wall_width, t_range, kB):
    """Linear T across the free interior (``t_bottom`` at ``z = w``, ``t_top`` at ``Lz - w``), returning
    through the wall slab on the gravity's closed-form ramp. Returns ``(kbt_field eV, T(z) K)``."""
    grid, box, z, u, _ = _z_axis(grid, box)
    t_top, t_bottom, w, Lz = float(t_top), float(t_bottom), float(wall_width), box[2]
    if min(t_top, t_bottom) <= 0.0 or not 0.0 < w < 0.5 * Lz:
        raise ValueError("temperatures must be > 0 K and 0 < wall_width < Lz/2")
    L_free = Lz - 2.0 * w
    G_in = u * (1.0 - Lz / (2.0 * w)) - (Lz / (2.0 * np.pi)) * np.sin(np.pi * u / w)
    s = np.where(np.abs(u) < w, (G_in - (w - 0.5 * Lz)) / L_free, (z - w) / L_free)
    t_prof = t_bottom + (t_top - t_bottom) * s
    if t_prof.min() <= 0.0:
        raise ValueError(f"the wall-slab blend drives T non-positive (min {t_prof.min():.1f} K)")
    _check_t_window("t_profile_walled", t_prof, t_range)
    return _kbt(grid, t_prof, kB), t_prof


def t_profile_mirror(grid, box, t_top, t_bottom, *, smooth_width, t_range, kB):
    """``t_top`` at the wrap, ``t_bottom`` at the mirror plane, on the gravity's own shape (no overshoot)."""
    grid, box, z, u, _ = _z_axis(grid, box)
    t_top, t_bottom, w, Lz = float(t_top), float(t_bottom), float(smooth_width), box[2]
    if min(t_top, t_bottom) <= 0.0 or not 0.0 < w < 0.25 * Lz:
        raise ValueError("temperatures must be > 0 K and 0 < smooth_width < Lz/4")
    G = _mirror_G(u, Lz, w)
    G_max = float(_mirror_G(np.array([0.0]), Lz, w)[0])
    G_min = float(_mirror_G(np.array([-0.5 * Lz]), Lz, w)[0])
    t_prof = t_top + (t_bottom - t_top) * ((G_max - G) / (G_max - G_min))
    _check_t_window("t_profile_mirror", t_prof, t_range)
    return _kbt(grid, t_prof, kB), t_prof


def t_profile_step(grid, box, t_flat, t_hot, frac, width, *, t_range, ramp_min_cells, kB):
    """Mirror only: ``t_flat`` over the top of the half column, one C1 raised-cosine step centred at depth
    ``frac * Lz/2`` of half-width ``width``, ``t_hot`` below; the step must fit and be resolved."""
    grid, box, z, u, dz = _z_axis(grid, box)
    t_flat, t_hot, frac, width = (float(v) for v in (t_flat, t_hot, frac, width))
    half = 0.5 * box[2]
    if min(t_flat, t_hot) <= 0.0 or not 0.0 < frac < 1.0 or not width > 0.0:
        raise ValueError("need t_flat, t_hot > 0 K, 0 < frac < 1, width > 0")
    d_lo, d_hi, tol = frac * half - width, frac * half + width, 1e-9 * box[2]
    problems = []
    if 2.0 * width < float(ramp_min_cells) * dz:
        problems.append(f"the ramp spans {2.0 * width / dz:.2f} cells, fewer than "
                        f"{float(ramp_min_cells):g}")
    if d_lo < -tol:
        problems.append(f"the step runs through the wrap (starts at depth {d_lo:g})")
    if d_hi > half + tol:
        problems.append(f"the step runs through the mirror plane (ends at {d_hi:g} > {half:g})")
    if problems:
        raise ValueError("; ".join(problems))
    d = np.abs((z + half) % box[2] - half)
    s = 0.5 * (1.0 - np.cos(np.pi * np.clip((d - d_lo) / (2.0 * width), 0.0, 1.0)))
    t_prof = np.where(s >= 1.0, t_hot, t_flat + (t_hot - t_flat) * s)
    _check_t_window("t_profile_step", t_prof, t_range)
    return _kbt(grid, t_prof, kB), t_prof


# -- initial state, diagnostics --

def uniform_ic(grid, rho_h, rho_he, ic_noise, seed) -> torch.Tensor:
    """Uniform state plus relative Gaussian noise, mean-free per species; ``(1, 2, *grid)`` float64."""
    rng = np.random.default_rng(int(seed))
    rho = np.empty((1, 2, *(int(g) for g in grid)), dtype=np.float64)
    for i, r0 in enumerate((float(rho_h), float(rho_he))):
        d = 0.0
        if ic_noise:
            d = rng.normal(0.0, float(ic_noise) * r0, size=tuple(int(g) for g in grid))
            d -= d.mean()
        rho[0, i] = r0 + d
    return torch.from_numpy(rho)


def _weighted_quantile(x, w, q):
    o = np.argsort(x, kind="stable")
    xs, cw = x[o], np.cumsum(w[o])
    i = int(np.searchsorted(cw, q * cw[-1], side="left"))
    return float(xs[min(i, len(xs) - 1)])


def rain_band(c_prof, interior):
    """Longest contiguous run of ``c > cbar + (q95 - cbar)/2`` (interior bins), ties to the low index;
    ``(-1, -1)`` when absent. The rule ``"mean_ref_v2"``."""
    c = np.asarray(c_prof, dtype=np.float64).ravel()
    m = np.asarray(interior, dtype=bool).ravel() & np.isfinite(c)
    w = np.ones_like(c)
    if not m.any():
        return (-1, -1)
    cbar = float((w[m] * c[m]).sum() / w[m].sum())
    hit = m & (c > cbar + 0.5 * (_weighted_quantile(c[m], w[m], 0.95) - cbar))
    best, best_len, i = (-1, -1), 0, 0
    while i < len(hit):
        if not hit[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(hit) and hit[j + 1]:
            j += 1
        if j - i + 1 > best_len:
            best, best_len = (i, j), j - i + 1
        i = j + 1
    return best


def frame_metrics(rho, z, interior, uniform_g, dV, domain) -> dict:
    """Per-frame diagnostics of one real field ``(2, Gx, Gy, Gz)``: profiles, ``Phi`` and the rain band over
    ``interior``, the He centre of mass over ``uniform_g``, trust-domain occupancy, masses, minima."""
    rho = np.asarray(rho, dtype=np.float64)
    tot = np.maximum(rho.sum(axis=0), 1e-9)
    c = rho[1] / tot
    ti = tot[:, :, interior]
    wgt = ti / ti.sum()
    ci = c[:, :, interior]
    cbar = (wgt * ci).sum()
    rhe = rho[1][:, :, uniform_g]
    occ = {}
    for name, m in (("interior", interior), ("wall", ~interior)):
        if not m.any():
            occ[name] = float("nan")
            continue
        pts = torch.from_numpy(np.ascontiguousarray(rho[:, :, :, m].reshape(2, -1).T))
        occ[name] = float(in_domain(pts, domain).numpy().mean())
    prof = rho.mean(axis=(1, 2))
    return dict(prof=prof, Phi=float((wgt * (ci - cbar) ** 2).sum()),
                z_he_com=float((rhe * z[uniform_g]).sum() / max(rhe.sum(), 1e-30)),
                mass=rho.sum(axis=(1, 2, 3)) * dV,
                occ_interior=occ["interior"], occ_wall=occ["wall"],
                rho_min=rho.min(axis=(1, 2, 3)),
                rain=rain_band(prof[1] / np.maximum(prof.sum(axis=0), 1e-9), interior))


# -- chunks: the previous chunk's last field, memory-mapped --

def mmap_fields(path, member="rho_fields.npy") -> np.ndarray:
    """One STORED ``.npy`` member of an uncompressed npz, memory-mapped; refuses anything else."""
    path = Path(path)
    with zipfile.ZipFile(path) as z:
        zi = z.getinfo(member)
        if zi.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"{path}: {member} is compressed and cannot be mapped")
    with open(path, "rb") as fh:
        fh.seek(zi.header_offset)
        head = fh.read(30)
        fh.seek(zi.header_offset + 30 + int.from_bytes(head[26:28], "little")
                + int.from_bytes(head[28:30], "little"))
        ver = np.lib.format.read_magic(fh)
        reader = {(1, 0): np.lib.format.read_array_header_1_0,
                  (2, 0): np.lib.format.read_array_header_2_0}.get(ver)
        if reader is None:
            raise ValueError(f"{path}: {member} is .npy version {ver}")
        shape, fortran, dtype = reader(fh)
        off = fh.tell()
    if fortran:
        raise ValueError(f"{path}: {member} is Fortran-ordered")
    return np.memmap(path, mode="r", offset=off, shape=shape, dtype=dtype)


def open_chunk(path) -> dict:
    """``params``, ``t``, ``t_fields`` and the mapped ``rho_fields`` of a column npz."""
    with np.load(path, allow_pickle=False) as z:
        meta = {k: z[k] for k in ("params", "t", "t_fields")}
    fields = mmap_fields(path)
    n = int(fields.shape[0]) if fields.ndim == 5 else 0
    if n == 0 or meta["t_fields"].size != n:
        raise ValueError(f"{path}: {n} saved fields, {meta['t_fields'].size} field times")
    return dict(params=json.loads(str(meta["params"])), t=meta["t"],
                t_fields=meta["t_fields"], fields=fields)


def _canon(v):
    return json.loads(json.dumps(v, default=repr))


def out_tag(p: dict) -> str:
    """The output-name suffix: state, gravity, geometry, profile, grid, noise, projection, ``m_stab``."""
    tag = f"_T{p['T']:g}_x{p['x_he']:g}_r{p['rho_tot']:g}_bo{p['bond']:g}"
    tag += (f"_mirror_ww{p['wall_width']:g}" if p["geometry"] == "mirror"
            else f"_wa{p['wall_amp']:g}_ww{p['wall_width']:g}")
    tp = p["t_profile"]
    if tp is not None and tp[0] == "linear":
        tag += f"_Tz{tp[1]:g}-{tp[2]:g}"
    elif tp is not None:
        tag += f"_Tstep{tp[1]:g}-{tp[2]:g}f{100.0 * tp[3]:g}w{tp[4]:g}"
    tag += "_g" + "x".join(str(g) for g in p["grid"]) + f"_dx{p['dx']:g}_icn{p['ic_noise']:g}"
    s = p["declaration"]["solver"]
    if p["noisy"]:
        tag += f"_sde_{s['noise_eval']}_ns{s['noise_scale']:g}_seed{p['seed']}"
    elif p["ic_noise"]:
        tag += f"_seed{p['seed']}"
    tag += "_sdom" if s["state_proj"] == "domain" else f"_sclamp{s['state_clamp']:g}"
    return tag + ("_mmax" if p["m_stab"] == "max" else "")


# -- the driver --

def column(system: System, ckpt, *, geometry: str, T: float, x_he: float,
           rho_tot: float, grid: Sequence[int], dx: float, bond: float,
           wall_amp: Optional[float], wall_width: float,
           t_profile: Optional[Sequence], ic_noise: float, seed: int,
           noisy: bool, t_end: float, dt: float, save_ps: float,
           field_stride: int, allow_t_edge: bool, resume_from: Optional[Path],
           chunk: Optional[int], device: str, out: Optional[Path] = None,
           declaration: Optional[Mapping[str, Any]] = None) -> Path:
    """One column run (or chunk); the noisy half runs ``defaults["column"]["noise"]``, the deterministic
    half ``System.noise.m_stab``. Writes ``column<tag>[_c<N>].npz`` with the published key set into
    ``<out or data/hhe/rollout>/<md5 of the configuration>[:12]`` and returns its path.
    ``t_profile``: ``None``, ``("linear", T_top, T_bottom)`` or ``("step", T_flat, T_hot, frac, width)``."""
    decl = declared_column(system, declaration)
    kB = float(system.constants["kB"])
    if geometry not in GEOMETRIES:
        raise ValueError(f"geometry must be one of {list(GEOMETRIES)}, got {geometry!r}")
    if geometry == "walled" and wall_amp is None:
        raise ValueError("the walled column needs wall_amp (in kBT)")
    if geometry == "mirror" and wall_amp is not None:
        raise ValueError("the mirror column has no wall: wall_amp must be None")
    if t_profile is not None:
        t_profile = [str(t_profile[0])] + [float(v) for v in t_profile[1:]]
        if t_profile[0] not in PROFILES or len(t_profile) != (3 if t_profile[0] == "linear" else 5):
            raise ValueError("t_profile must be ('linear', T_top, T_bottom) or "
                             "('step', T_flat, T_hot, frac, width)")
        if t_profile[0] == "step" and geometry != "mirror":
            raise ValueError("the step profile is defined on the mirror geometry only")
    if int(field_stride) < 1 or min(dt, t_end, save_ps, dx) <= 0.0:
        raise ValueError("field_stride >= 1 and dt, t_end, save_ps, dx > 0 are required")
    grid = [int(g) for g in grid]
    box = tuple(g * float(dx) for g in grid)
    rho_h, rho_he = float(rho_tot) * (1.0 - float(x_he)), float(rho_tot) * float(x_he)
    domain = trust_domain(system)
    if not bool(in_domain(torch.tensor([[rho_h, rho_he]], dtype=torch.float64), domain)[0]):
        raise ValueError(f"IC rho_H={rho_h:.4g}, rho_He={rho_he:.4g} is outside the trust domain")
    lo, hi = (float(v) for v in decl["T_window"])
    margin = float(decl["T_margin"])
    if t_profile is not None:
        bad = [t for t in t_profile[1:3] if not lo + margin <= t <= hi - margin]
        if bad and not allow_t_edge:
            raise ValueError(f"profile endpoints {bad} K are within {margin:g} K of the "
                             f"window [{lo:g}, {hi:g}] K or outside it (allow_t_edge to run)")
    if not lo + margin <= float(T) <= hi - margin:
        warnings.warn(f"T {T:g} K (the potential's kBT reference) is within {margin:g} K "
                      f"of the window [{lo:g}, {hi:g}] K or outside it")
    if t_profile is not None and t_profile[0] == "step" and abs(float(T) - t_profile[1]) > margin:
        warnings.warn(f"T {T:g} K sets g_eff through the Bond number, but the flat region is at "
                      f"{t_profile[1]:g} K")
    n_steps = int(round(t_end / dt))
    save_every = max(1, int(round(save_ps / dt)))
    if (chunk is not None or resume_from is not None) and n_steps % save_every:
        raise ValueError(f"a chunked run needs save_ps to divide t_end: {n_steps} steps, "
                         f"a save every {save_every}")
    path = resolve_checkpoint(system, ckpt)
    md5 = _file_md5(path)
    m_stab = check_m_stab(decl["noise"].m_stab if noisy else declared_noise(system)["m_stab"])
    params = _canon(dict(
        geometry=geometry, grid=grid, dx=float(dx), bond=float(bond), wall_amp=wall_amp,
        wall_width=float(wall_width), T=float(T), x_he=float(x_he), rho_tot=float(rho_tot),
        t_profile=t_profile, noisy=bool(noisy), dt=float(dt), ckpt_md5=md5,
        declaration=decl, m_stab=m_stab, ic_noise=float(ic_noise), seed=int(seed),
        t_end=float(t_end), save_ps=float(save_ps), field_stride=int(field_stride),
        allow_t_edge=bool(allow_t_edge), resume_from=None, chunk=chunk,
        t_start=0.0))
    out_dir = output_dir(system, "column", md5,
                         {k: params[k] for k in COMPAT_KEYS}, out)
    chunked = chunk is not None or resume_from is not None
    src = None
    if resume_from is not None:
        src_path = Path(resume_from).resolve()
        src = open_chunk(src_path)
        bad = [(k, src["params"].get(k), params[k]) for k in COMPAT_KEYS
               if src["params"].get(k) != params[k]]
        if bad:
            raise ValueError(f"{src_path} was written with a different configuration: "
                             + "; ".join(f"{k}: {a!r} != {b!r}" for k, a, b in bad))
        if chunk is None:
            prev = src["params"].get("chunk")
            chunk = 1 if prev is None else int(prev) + 1
        t_start = float(src["t_fields"][-1])
        if src["t"].size and float(src["t"][-1]) - t_start > 1e-9 * max(1.0, abs(t_start)):
            raise ValueError(f"{src_path}: rows run to {float(src['t'][-1]):g} ps past the "
                             f"last saved field at {t_start:g} ps")
        params.update(resume_from=str(src_path), chunk=int(chunk), t_start=t_start)
    if chunk is not None and int(chunk) < 0:
        raise ValueError(f"chunk must be >= 0, got {chunk}")
    seed_eff = int(seed) + int(decl["seed_chunk_stride"]) * int(chunk or 0)
    params["seed_effective"] = seed_eff
    target = out_dir / f"column{out_tag(params)}{'' if chunk is None else f'_c{int(chunk)}'}.npz"
    if src is not None and target.resolve() == Path(params["resume_from"]):
        raise ValueError(f"this run would overwrite its own source {target}")

    if geometry == "mirror":
        v_ext, interior, info = mirror_vext(grid, box, T, bond=bond, smooth_width=wall_width,
                                            kB=kB, masses=decl["masses"])
    else:
        v_ext, interior, info = walled_vext(
            grid, box, T, bond=bond, wall_amp_kbt=wall_amp, wall_width=wall_width,
            quarantine_thresh=decl["quarantine_thresh"], kB=kB, masses=decl["masses"])
    uniform_g = info["uniform_g_mask"]
    kbt_field, t_prof, T_ref = None, np.zeros(0), float(T)
    t_range = None if allow_t_edge else (lo, hi)
    if t_profile is not None:
        if t_profile[0] == "step":
            kbt_field, t_prof = t_profile_step(grid, box, *t_profile[1:], t_range=t_range,
                                               ramp_min_cells=decl["ramp_min_cells"], kB=kB)
        elif geometry == "mirror":
            kbt_field, t_prof = t_profile_mirror(grid, box, *t_profile[1:],
                                                 smooth_width=wall_width, t_range=t_range, kB=kB)
        else:
            kbt_field, t_prof = t_profile_walled(grid, box, *t_profile[1:],
                                                 wall_width=wall_width, t_range=t_range, kB=kB)
        T_ref = float(kbt_field.mean(dtype=torch.float64)) / kB
        kbt_field = kbt_field.to(device)

    if src is not None:
        rho0 = torch.from_numpy(np.array(src["fields"][-1], dtype=np.float32))[None]
        if tuple(rho0.shape[1:]) != (2, *grid):
            raise ValueError(f"the source's last field has shape {tuple(rho0.shape[1:])}")
    else:
        rho0 = uniform_ic(grid, rho_h, rho_he, ic_noise, seed)
    src = None
    model = load_model(system, path).to(device)
    ops = SpectralOps(tuple(grid), system.n_species, nyquist_mask=model.ops.nyquist_mask)
    n_cells = grid[0] * grid[1] * grid[2]
    rho_hat0 = (ops.rfft(rho0.to(device)) / n_cells).to(torch.complex64)
    s = decl["solver"]
    generator = noise = None
    if noisy:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed_eff)
        noise = dict(noise_for(system, {"solver": s}, T_ref), noise_mode=decl["noise"].mode)
        if decl["noise"].mode != "gaussian":
            noise["sigma_noise"] = None
    traj = rollout_imex(
        model, rho_hat0, torch.tensor(box, dtype=torch.float32), T_ref, dt, n_steps,
        kB=kB, m_stab=m_stab, state_proj=s["state_proj"], state_clamp=s["state_clamp"],
        domain=domain, mass_restore=s["mass_restore"], clamp_rho=s["clamp_rho"],
        noise=noise, save_every=save_every, generator=generator,
        v_ext=v_ext.to(device), kbt_field=kbt_field)

    dV = float(np.prod(box)) / n_cells
    f0 = 1 if params["resume_from"] is not None else 0
    rows, fields, t_idx = [], [], []
    for f in range(f0, traj.shape[0]):
        rho = ops.irfft(traj[f:f + 1].to(device) * n_cells)[0].cpu().numpy()
        rows.append(frame_metrics(rho, info["z"], interior, uniform_g, dV, domain))
        if f % int(field_stride) == 0 or (chunked and f == traj.shape[0] - 1):
            fields.append(rho.astype(np.float32))
            t_idx.append(f)
    if not rows:
        raise ValueError("the chunk saved no new frame (t_end < save_ps)")

    def times(idx):
        return params["t_start"] + np.asarray(idx, dtype=np.float64) * float(save_ps)

    prof = np.stack([r["prof"] for r in rows])
    mass = np.stack([r["mass"] for r in rows])
    np.savez(
        target, t=times(np.arange(f0, traj.shape[0])), prof_H=prof[:, 0], prof_He=prof[:, 1],
        Phi=np.array([r["Phi"] for r in rows]),
        z_he_com=np.array([r["z_he_com"] for r in rows]),
        mass_H=mass[:, 0], mass_He=mass[:, 1],
        occ_interior=np.array([r["occ_interior"] for r in rows]),
        occ_wall=np.array([r["occ_wall"] for r in rows]),
        rho_min=np.stack([r["rho_min"] for r in rows]),
        rain_band=np.array([r["rain"] for r in rows], dtype=np.int64),
        rain_rule=decl["rain_rule"], t_profile=t_prof, t_fields=times(t_idx),
        rho_fields=np.stack(fields) if fields else np.zeros(0),
        interior_mask=interior, uniform_g_mask=uniform_g, z=info["z"],
        V_wall=info["V_wall"], G=info["G"], g_eff=info["g_eff"], bond=info["bond"],
        L_free=info["L_free"], wall_width=info["wall_width"],
        wall_amp_kbt=info["wall_amp_kbt"], kBT=info["kBT"], dz=info["dz"], T_K=float(T),
        sigma=float(system.defaults["sigma"]), box=np.array(box), grid=np.array(grid),
        params=json.dumps(params))
    manifest = out_dir / "MANIFEST.json"
    outputs = json.loads(manifest.read_text())["outputs"] if manifest.is_file() else []
    write_manifest(out_dir, system, "column", path, md5,
                   {k: params[k] for k in COMPAT_KEYS},
                   sorted(set(outputs) | {target.name}))
    written = json.loads(manifest.read_text())
    written["noise"] = ({"noise_mode": noise["noise_mode"], "sigma_noise": noise["sigma_noise"],
                         "m_stab": m_stab} if noisy else {"noise_mode": None, "m_stab": m_stab})
    manifest.write_text(json.dumps(written, indent=1, default=repr))
    return target


def main(argv=None) -> int:
    """``python -m experiments.hhe.column``: every flag required except the optional ones."""
    from aipf.system import load
    ap = argparse.ArgumentParser(prog="python -m experiments.hhe.column")
    req = dict(required=True)
    ap.add_argument("--ckpt", **req, help="'published' (the declared checkpoint, md5-checked) or a path")
    ap.add_argument("--geometry", choices=GEOMETRIES, **req)
    for flag in ("--T", "--x-he", "--rho-tot", "--dx", "--bond", "--wall-width",
                 "--ic-noise", "--t-end", "--dt", "--save-ps"):
        ap.add_argument(flag, type=float, **req)
    ap.add_argument("--wall-amp", type=float, default=None, help="walled geometry only")
    ap.add_argument("--grid", **req, help="Gx,Gy,Gz")
    ap.add_argument("--seed", type=int, **req)
    ap.add_argument("--field-stride", type=int, **req)
    ap.add_argument("--noisy", choices=("yes", "no"), **req)
    ap.add_argument("--t-profile", default=None,
                    help="linear,T_top,T_bottom or step,T_flat,T_hot,frac,width")
    ap.add_argument("--allow-t-edge", action="store_true")
    ap.add_argument("--resume-from", default=None)
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--device", **req)
    ap.add_argument("--out", **req)
    a = ap.parse_args(argv)
    tp = None if a.t_profile is None else a.t_profile.split(",")
    target = column(load("hhe"), a.ckpt, geometry=a.geometry, T=a.T, x_he=a.x_he,
                    rho_tot=a.rho_tot, grid=[int(g) for g in a.grid.split(",")], dx=a.dx,
                    bond=a.bond, wall_amp=a.wall_amp, wall_width=a.wall_width,
                    t_profile=tp, ic_noise=a.ic_noise, seed=a.seed,
                    noisy=a.noisy == "yes", t_end=a.t_end, dt=a.dt, save_ps=a.save_ps,
                    field_stride=a.field_stride, allow_t_edge=a.allow_t_edge,
                    resume_from=a.resume_from, chunk=a.chunk, device=a.device,
                    out=Path(a.out))
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
