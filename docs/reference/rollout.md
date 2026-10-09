# Rollouts

`aipf rollout` integrates a trained functional from frame 0 of an archived MD run and compares the
two. Two drivers, `aipf.rollout.spinodal.spinodal` and `aipf.rollout.slab.slab`:

| driver | from | compares |
|---|---|---|
| `spinodal` | a homogeneous cube quenched into the unstable region | `Phi(t)`, the density-weighted variance of `c = rho_x / sum rho`; the radially binned `S_cc(k, t)` on integer wavenumbers; the domain size `L(t) = L_box / k1` |
| `slab` | a phase-separated slab | the profile `c(z)` along the declared axis, its two plateaus, the 10-90 interface width, `Phi(t)` |

Both sides go through one reconstruction: the archived modes are scattered onto the declared grid
and filtered by `exp(-k^2 sigma^2 / 2)` with `defaults["sigma"]`.

    spinodal(system, ckpt, *, run, seeds, t_end, dt, save_ps, device, out=None, declaration=None,
             precision="fp32") -> Path
    slab(system, ckpt, *, run, seeds, t_end, dt, save_ps, device, out=None, declaration=None,
         precision="fp32") -> Path

| argument | meaning |
|---|---|
| `ckpt` | `"published"` (the system's checkpoint, digest-checked) or a path; a file `aipf train` wrote (tagged `config_schema`) loads from its `model_state_dict` |
| `run` | the archived run's directory under the driver's `modes_tree` |
| `seeds` | one noisy rollout per seed; empty: one deterministic rollout |
| `t_end`, `dt`, `save_ps` | the simulated time, the step and the saving interval, in ps |
| `device` | a torch device |
| `out` | the output root; `None` is `<data>/<system>/rollout/` |
| `declaration` | a block of the shape of `defaults["rollout"]` that replaces it |
| `precision` | `"fp32"` (the default, the published rollouts) or `"fp64"` ([Precision](#precision)) |

The output directory is `<out>/<md5[:12]>/`, the digest of the driver, the checkpoint's md5 and the
whole request, so a repeated request lands in the same place. `precision` enters the request only when
it is not `fp32`: an fp32 run keeps the directory and manifest it always had, an fp64 run gets its own.

## The declaration

`defaults["rollout"]` has three parts; each must carry exactly its keys.

`solver`:

| key | admissible | meaning |
|---|---|---|
| `clamp_rho` | float | the floor under the density `mu` and `M` are evaluated at |
| `state_proj` | `floor`, `domain` | project the state onto `rho >= state_clamp`, or onto the system's trust trapezoid |
| `state_clamp` | float | the floor of `floor`; the lower bound under `domain` with `headroom` |
| `mass_restore` | `shift`, `headroom` | after the projection, write the `k = 0` mode back by a uniform shift, or restore it without taking a cell below the bound (`aipf.solve.project_state`) |
| `noise_eval` | `ito`, `midpoint`, `kinetic` | where the noise amplitude is evaluated: at `rho^n`, or at a predictor advanced by half or all of the step |
| `noise_scale` | float | multiplies the noise amplitude |
| `predictor_floor` | float | the floor of that predictor's density |

`archive` (how the mode archive spells its keys): `file_name`, `amplitudes`,
`amplitudes_channel_axis`, `labels`, `box`, `temperature`, `frame_interval`, `quality_file`,
`quality_key`. A quality file, when present, gives the last valid frame under `quality_key`.

`spinodal`: `modes_tree` (relative to the raw root), `grid` (three ints), `field_stride` (frames
between stored fields). `slab`: the same plus `profile_axis` (0, 1 or 2).

The noise comes from `System.noise` (`mode` and `m_stab`, [system.md](system.md#noise)), with
amplitude `kB T`; the density edges from `System.trust_domain`. A system that declares no
`rollout` block, no noise or no trust domain is refused, the message listing every missing key.

## The scheme

`aipf.rollout.imex.rollout_imex` is a semi-implicit step with a frozen operator
`A = I + dt k^2 M_s H(k)`, `H = Hess f_loc(rho_bar) + W(k)`, `M_s` the mobility at the mean density
(`m_stab = "mean"`) or at the initial field's largest-norm cell (`"max"`):

    rho^{n+1} = A^-1 (rho^n + dt F(rho^n) + dt k^2 M_s H rho^n + dt n_hat)

then the Nyquist modes are made Hermitian and the state is projected. A model with a
`stabilizer_mobility(rho, T)` method sets `M_s` itself: it is called once, on the real-space field at
`t = 0` `(1, n, Gx, Gy, Gz)`, unclamped, and `T` `(1,)`, and its `(n, n)` return is refused unless finite,
symmetric and positive semi-definite to a relative `1e-6`. `m_stab` is then left undeclared, and a
declared one is refused, so the drivers that pass the system's `Noise.m_stab` refuse such a model.
`aipf.rollout.imex` logs which of the three set `M_s`: the hook at `INFO`, `mean` and `max` at `DEBUG`. The conserved noise is
`n_hat = G(k) i k . zeta_hat`, `zeta = noise_scale sqrt(2 kB T / (dV dt)) L w`, `L L^T = M`, `w`
standard normal per cell, direction and channel, with
`G(k) = exp(-k^2 sigma^2 / 2)` (`gaussian`) or 1 (`none`). It reads the model's `kernel.w_hat` and
`f_local`, so it runs pair-kernel functionals only. `aipf.solve` holds the explicit integrators
(`euler`, `heun`, `sde`) for any functional, and their state projection
`aipf.solve.project_state(rho_hat, ops, floor, *, state_proj, domain=None, lo, hi)`, which keeps the
`k = 0` mode exactly. Its `state_proj` is one of `aipf.solve.STATE_PROJ_MODES`:

| `state_proj` | projects onto | needs |
|---|---|---|
| `floor` | `rho >= floor` per cell (`floor <= 0`: no projection) | `floor` |
| `box` | `lo <= rho <= hi` per cell | `lo`, `hi` |
| `domain` | the system's trust trapezoid | a `TrustDomain` and `lo > 0` for the mass restore |

The semi-implicit scheme above takes `floor` and `domain` only.

Both schemes run a two-dimensional model (`model.ops.ndim == 2`): `rho_hat` `(1, n, Gx, Gy//2+1)` in a
box `(2,)` for the semi-implicit one, `(B, 2)` boxes for the explicit ones, `kbt_field` `(Gx, Gy)`,
`v_ext` `(n, Gx, Gy)`, the noise drawn `(B, n, 2, Gx, Gy)` and `stabilizer_mobility` handed
`(1, n, Gx, Gy)`. Under noise the call declares `depth`, the cell's extent along the averaged axis
(`dV = dA * depth`; 1.0 for areal densities, `Lz` for volumetric ones), and a three-dimensional call
declares none. See [functional.md](functional.md#two-dimensions).

## Precision

Both precisions run the same scheme; which one is chosen by the dtype of the state and the model
(`aipf.solve.precision.working_dtypes`), never by a separate argument of `rollout_imex`:

| state | model | runs |
|---|---|---|
| `complex64` | float32 | the float32 scheme the published rollouts ran, bit for bit (`tests/golden`) |
| `complex128` | float64 (`model.double()`) | float64 / complex128 end to end |
| `complex128` | float32 | refused: cast the model with `model.double()`, or the state with `rho_hat.to(torch.complex64)` |
| `complex64` | float64 | refused: cast the state with `rho_hat.to(torch.complex128)`, or the model with `model.float()` |

A model's precision is that of its parameters and of its operator set's mode buffer `ops.NX` (other
buffers are not read; a float32 model may carry a float64 constant). On the float64 path every tensor
the scheme makes is float64 or complex128: the operator set and its wavenumbers, `k^2`, `A^-1`, `M_s H`
and the predictor's pair, the local Hessian, `M_s` (hook, `mean` or `max`), `T`, the box, `kbt_field`
and `v_ext` (cast from whatever they are given in; give the box in float64 when its lengths are not
float32 numbers), the noise filter, draws and divergence, the Hermitian repair and the projections.
`tests/unit/test_rollout_float64.py` checks it against an independent float64 numpy step to `1e-12`
and audits every tensor a torch function returns during the rollout.

The wavenumbers come from exact integer mode indices. The float32 buffer `NX = fftfreq(G) * G` is an
integer only to about `4e-8 G` (`3.8e-6` at `G = 100`, `3e-5` at `G = 1000`), and `model.double()` is a
plain cast that keeps that error. `SpectralOps(..., dtype=torch.float64)` (and `SpectralOps2D`,
`make_ops`) builds every buffer in float64 with `NX` rounded to integers; the scheme builds its own
operator set that way, and for the duration of the call `aipf.spectral.exact_mode_indices(model)` has
the model's float64 operator sets read their `NX`, `NY`, `NZ` rounded and its `OpsCache` build other grids
in float64. Everything is put back on exit: the model, its state dict and its cache are the ones it had.
A float32 set is never touched, and a float64 set has the float32 one's state-dict keys and shapes.

A float64 noisy run draws float64 normals: the same seed is not the float32 run's random stream, and the
two noisy trajectories are not comparable draw for draw.

The explicit integrators (`aipf.solve`) follow the model the same way: an operator set they build takes
the dtype of `model.ops` (float64 with exact indices for a float64 model), a float64 state on float64
operators reads `boxes` and `T` in float64, and `rollout_deterministic` / `rollout_sde` run inside
`exact_mode_indices`.

The drivers take `precision="fp64"` (and `aipf rollout --precision fp64`): the loaded model is cast in
place (`model.double()`), and the initial state and box are cast to float64 before the solver. The
initial state is the archive's modes as read (scattered `complex64`, then filtered), cast; only the
rollout itself is float64.
`aipf.rollout.fdt.run_one` takes it too, and builds its homogeneous state and `H(k)` in float64.

## Outputs

`spinodal_<run>_<det|s<seed>>.npz`: `t_model`, `Phi_model`, `L_model`, `Sk_model`, `t_md`, `Phi_md`,
`L_md`, `Sk_md`, `box_L`, `T_K`, `t_fields`, `c_model`, `c_md` (fields every `field_stride` frames),
`rho_hat_final`.

`<run>_<det|s<seed>>.npz` (slab): `t_model`, `t_md`, `prof_model`, `prof_md`, `x_poor_model`,
`x_rich_model`, `x_poor_md`, `x_rich_md`, `width_model`, `width_md`, `Phi_model`, `Phi_md`, `box_L`,
`T_K`, `t_fields`, `c_model`, `c_md`, `rho_hat_final`.

`MANIFEST.json`: system, driver, checkpoint and md5, the noise declaration, the request (with the
declaration used, and `precision` for an fp64 run) and the files written. An fp64 run's
`rho_hat_final` is `complex128`.

## Example

    aipf rollout spinodal --system hhe --ckpt published --run cube_x0.50_T07000 --seeds \
        --t-end 0.2 --dt 1e-4 --save-ps 0.02 --device cuda --out data

needs the H/He raw root (`fields/modes_800GPa/cube_x0.50_T07000/modes.npz`) and writes
`spinodal_cube_x0.50_T07000_det.npz` and `MANIFEST.json`.

## The H/He rain column

`experiments/hhe/column.py` is a system-specific driver: a column of H/He under gravity, with walls
(`walled`) or a mirror plane (`mirror`), declared in `defaults["column"]` (its own `Noise`, the
gravity masses, the trained temperature window, the solver block). Every flag is required:

    python -m experiments.hhe.column --ckpt published --geometry {walled,mirror} --T T --x-he X
        --rho-tot RHO --dx DX --bond BOND --wall-width W --ic-noise A --t-end T --dt DT --save-ps S
        --grid GX,GY,GZ --seed N --field-stride K --noisy {yes,no} --device DEV --out DIR
        [--wall-amp A] [--t-profile linear,T_top,T_bottom | step,T_flat,T_hot,frac,width]
        [--allow-t-edge] [--resume-from FILE] [--chunk N]

run from the repository root; `--help` describes each flag.
