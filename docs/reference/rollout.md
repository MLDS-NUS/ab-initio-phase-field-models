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
`aipf.rollout.imex` logs which source set `M_s`: the hook at `INFO`, `mean`, `max` and `bound` at `DEBUG`. The conserved noise is
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

### The stabiliser: `m_stab="bound"`, refreshes and the log

Three keyword arguments of `rollout_imex` extend how `M_s` is frozen. Each is off by default, and a call
that leaves them off is the scheme above, bit for bit. They are rollout options: `Noise.m_stab` still
declares `mean` or `max` only, and the drivers and the CLI pass neither these arguments nor `"bound"`.

| `M_s` source | frozen at |
|---|---|
| `m_stab="mean"` | `M(rho_bar)`, the mobility at the state's `k = 0` mode |
| `m_stab="max"` | the cell whose mobility has the largest eigenvalue |
| `m_stab="bound"` | `bound_margin * max(r, 1) * M(rho_bar)`, an upper bound of every cell's mobility |
| `stabilizer_mobility` hook | whatever the model returns (`m_stab` undeclared) |

**Why a bound.** In the stiff limit (`dt k^2 M H` much larger than 1 at the grid scale), a mode of a cell where
the mobility is `M_loc` and the Hessian `H_loc` is advanced by the frozen-coefficient amplification
`G -> I - (M_s H_bar)^-1 M_loc H_loc`. A mode the PDE damps is over-corrected (`G < -1`, a grid-scale blow-up
the explicit scheme does not show) wherever the largest generalised eigenvalue of `M_s^-1 M_loc` exceeds 2.
With a composition-dependent mobility, `M(rho_bar)` can sit below a cell's mobility by more than that, and
`"max"` picks the cell with the largest mobility in norm, not the one whose ratio against `M_s` is largest.

**`"bound"`.** `r` is the largest, over the grid cells, of `lambda_max(M_bar^-1/2 M(rho(x)) M_bar^-1/2)`, with
`M_bar = M(rho_bar)` built exactly as `"mean"` builds it and `M(rho(x))` the model's `mobility` on the real field
(the field the hook and `"max"` read; under `clamp_rho`, clamped as they are). It is computed in float64 whatever
the run's precision: `eigh` of the symmetrised `M_bar`, its eigenvalues floored at `1e-12` times the largest,
`W = V diag(lambda^-1/2) V^T`, and `eigvalsh` of the symmetrised `W M(rho(x)) W` per cell. `r` is clamped at 1
from below, so `"bound"` is never less damped than `"mean"` in any direction, and when every cell is below
`M_bar` the run is the `"mean"` run. When the floor raises an eigenvalue of `M_bar` (a singular or very
ill-conditioned `M_bar`), `M_s` is built from the floored matrix `V diag(lambda_floored) V^T`, so the bound holds
along `M_bar`'s null directions as well; otherwise `M_s` is `bound_margin * max(r, 1)` times the very `M_bar`
`"mean"` uses, the product taken in float64 and cast back to the run's dtype. Every cell then satisfies
`M(rho(x)) <= M_s / bound_margin` in the Loewner order, so the stiff-limit ratio is at most `1 / bound_margin`
(to the rounding of `M_s` in a float32 run). `bound_margin` is `None` (1.0) or a finite number `>= 1`, and is
refused with any other source. A non-finite `M_bar`, cell mobility, `r` or `M_s` is refused, naming which, and
so is an `M_bar` with no positive eigenvalue. A model with a `stabilizer_mobility` hook refuses `"bound"`, as it
refuses any declared `m_stab`.

The bound is only as good as `model.mobility`: it holds when `model.mobility(rho)` bounds, in the Loewner order,
the mobility the model's forward pass applies. A model whose drift uses a different mobility (for example a
gradient-dependent one) must make `mobility(rho)` an upper bound of it for `"bound"` to mean what it says.

**`refresh_every`.** `None` freezes `M_s` once, at `t = 0`. An int `N >= 1` re-freezes it from the current state
after steps `N, 2N, ...` (before the next step is taken), from the same source (hook, `mean`, `max` or `bound`),
and rebuilds `A^-1`, `M_s H` and, under noise, the predictor's pair exactly as a new call would. The local
Hessian at `rho_bar` stays frozen, as `rho_bar` is conserved; it is rebuilt only if the state's `k = 0` mode has
moved in its last bit. The operator set, the box, the noise filter and amplitude, `kbt_field`, `v_ext` and the
random stream carry on. A run with `refresh_every=N` is, bit for bit, `ceil(n_steps / N)` calls of `N` steps,
each from the previous call's last state with the same generator, keeping the steps the single call's
`save_every` saves. A `"mean"` refresh rebuilds the same operators. The cost of a refresh is dominated by the
operator setup (`A^-1` over the spectrum), not by the eigenproblem; when `r` grows slowly, refreshing every few
steps with a margin somewhat above 1 (for example `refresh_every=5`, `bound_margin=1.25`) is a reasonable
economy.

**`stabiliser_log`.** A list passed here receives one dict per freeze: `step` (the steps taken when it froze) and
`source` (`"hook"`, `"mean"`, `"max"` or `"bound"`); for `"bound"` also `r_raw`, `r_used` (`max(r_raw, 1)`),
`margin`, `cell` (the grid index of the largest ratio), `ratio_min`, `ratio_mean`, `ratio_max` (the cells'
ratios against `M_bar`), `cond` (`M_bar`'s condition number, `inf` if it is not positive definite), `floored`
and `ratio_after` (the cells' largest ratio against `M_s`, `r_raw / (bound_margin * r_used)` up to rounding).
With `None` nothing extra is computed.

`aipf.rollout.bound_ratio(model, rho_hat, T, *, clamp_rho=None)` computes the same quantities from one state
without rolling out: `r_raw`, `r_used`, `cell`, `ratios` (float64 on the grid), `M_bar`, `cond`, `floored` and
`M_s` at `bound_margin = 1`. A caller can check the states of a `"mean"` run with it: where `r_raw > 2` at a stiff
`dt`, `"mean"` over-corrects. `aipf.rollout.imex.generalised_ratios(M_bar, cells)` is the per-cell ratio.
`tests/unit/test_rollout_bound_stabiliser.py` checks `r` against `scipy.linalg.eigh(M_cell, M_bar)`, a singular
`M_bar`, the refresh against the chained calls and a stiff case where `"mean"` blows up and `"bound"` does not.

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
and the predictor's pair, the local Hessian, `M_s` (hook, `mean`, `max` or `bound`, refreshed or not), `T`,
the box, `kbt_field` and `v_ext` (cast from whatever they are given in; give the box in float64 when its lengths are not
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

The explicit integrators (`aipf.solve`) are unchanged by this option: they take no precision and run
exactly as before for every dtype, with their own float32 operator sets.

The drivers take `precision="fp64"` (and `aipf rollout --precision fp64`): the solver is handed a float64
copy of the model (`copy.deepcopy`, then cast; the caller's model keeps its dtype, so a later fp32 call
with it runs as before), and the initial state and box are cast to float64. The
initial state is the archive's modes as read (scattered `complex64`, then filtered), cast; only the
rollout itself is float64.
`aipf.rollout.fdt.run_one` takes it too, on a float64 copy of the model, and builds its homogeneous
state and `H(k)` in float64; `hessian_of_k` on its own evaluates in float32 unless told otherwise.

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
