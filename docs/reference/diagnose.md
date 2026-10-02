# Diagnosis

`aipf diagnose` and `aipf.diagnose.run.run` load one checkpoint into the system's declared
functional (float64) and run the stages asked for. Every threshold and grid a stage reads is
declared in the system's `defaults["diagnose"]`; nothing in the driver has a default.

    run(system, ckpt, *, stages, out=None, pressures=(), T_grid=None, override_declared=False,
        **declared) -> Path

`ckpt` is a file path or the system's `Checkpoint` (digest-verified). `declared` is the system's
`defaults["diagnose"]` block, forwarded unchanged. The output directory is
`<out>/<md5[:12]>/`, `out` defaulting to `<data>/<system>/diagnose/`, so one checkpoint always lands
in one place. Every run writes `MANIFEST.json`: the system, the checkpoint and its md5, the stages,
pressures and temperature grid used (with `T_grid_source`: `declared`, `command line`, `override` or
`none`), the files written and the declared block.

## Stages

| stage | systems | needs | writes |
|---|---|---|---|
| `phase_diagram` | two channels with measured equation-of-state manifolds (`hhe`, `feb`) | `--pressure` (each a key of `eos_csvs`) and a T grid | `P<P>GPa/phase_diagram.npz` per pressure, `summary.csv` |
| `stability_map` | two channels, pair-kernel functional, a declared `stability_map` block (`feb`) | `--pressure` and a T grid | `stability_map.npz` |
| `dome` | a declared `dome` block (`hhe`) | nothing: the grids are declared | `dome_Pgrid.npz` |
| `tc` | as `dome`, which it runs | nothing | `dome_Pgrid.npz`, `tc.json` |
| `kappa` | a pair-kernel functional with `kernel_n_quad` (`hhe`, `feb`) | nothing | `kappa.json` |
| `one_field_phase_diagram` | one channel (`lj`, its variants) | the declared `one_field["T_grid"]`; a flag must equal it, or `--override-declared` | `one_field_phase_diagram.npz` |

A stage asked for without what it needs is refused before the checkpoint is loaded.

### Outputs

| file | keys |
|---|---|
| `phase_diagram.npz` | `T`; `bin_lo`, `bin_hi` (the declared binodal route under the lever anchor `x_bar`); `bin1d_lo`, `bin1d_hi` (the widest tie overlapping the concave stretch); `spmat_lo`, `spmat_hi` (where `d^2 g / dx^2 < 0`); `Tc_fit`, `w0` (the apex fit); `iso_x`, `iso_s` (the isobar's density scale on `iso_x_grid`); `isobar_P_GPa`; `isobar = "model"`. NaN where there is no gap |
| `summary.csv` | per pressure: `P_GPa`, `T_c_K` (highest two-phase T on the grid), `T_c_fit_K`, `T_mid_K`, `bin_lo_mid`, `bin_hi_mid`, `max_abs_iso_s_minus_1` |
| `stability_map.npz` | `T`, `x_int`, and per pressure `lam_P<P>` (smallest eigenvalue of `H_tot(0) = d^2 f_loc / d rho^2 + W(0)`) and `Gamma_P<P>` (`x0 x1 / S_cc(0)`), on (T, x) |
| `dome_Pgrid.npz` | `T`, `P_grid`, `bin_lo`, `bin_hi` (P x T), `Tc_fit`, `w0`, `T_last`, `anchor_P` (the manifold each pressure used) |
| `tc.json` | `method`, `P`, `Tc_fit`, `w0`, `T_last` |
| `kappa.json` | `kappa_eff`, an `n x n` matrix ([functional.md](functional.md#nonlocal_kernel)) |
| `one_field_phase_diagram.npz` | `mu_roots`: `T`, `binodal_L`, `binodal_R`, `spinodal_L`, `spinodal_R`, `Tc_exact`, `mu_w0`, `ising_T_max`, `ising_B`, `ising_T_c`, `ising_beta`. `bulk_minima`: `T_grid`, `binodal_lo`, `binodal_hi`, `spinodal_lo`, `spinodal_hi`, `Tc` |

## Temperature grids

A grid is `(lo, hi, step)` or `{"lo", "hi", "step"}` (`np.arange(lo, hi + 1e-9, step)`), or
`{"lo", "hi", "n"}` (`np.linspace(lo, hi, n)`). On the command line, `--T-grid LO,HI,STEP` and
`--T-grid-n LO,HI,N`.

## The isobar

A two-channel system is diagnosed along the model's own isobars. The measured equation of state
gives a density `n_manifold(x, T)` at the manifold's pressure; the stage finds the scale `s(x)` with
`P_model(s n_manifold(x), T) = P`, so `n(x) = s(x) n_manifold(x)`, and builds
`g(x) = (f + P) / n`, whose lower convex hull gives the coexisting compositions.

`model_pressure(model, rho, T, *, route)`: `route` is `protocol` (the model's own pressure,
`aipf.train.pressure.pressure_from_model`) or `closed_form` (`mu_local . rho - f_local + (1/2)
rho^T W(0) rho`, pair-kernel functionals).

    isobar_scale(model, P, x, T, *, manifold, pressure_unit, poly_degree, s_lo, s_hi, n_scan,
                 n_iter, n_species, pressure_route) -> s(x)

| parameter | meaning |
|---|---|
| `model` | the functional |
| `P` | the isobar, in the unit of which one `pressure_unit` is the model's energy per volume |
| `x` | the channel-1 fractions along the path |
| `T` | the temperature |
| `manifold` | the measured `n(x, T)` table the scale multiplies (`manifolds(system, declared)[P]`) |
| `pressure_unit` | one unit of `P` in the model's energy-per-volume unit (`1 / 160.2176` eV/A^3 per GPa) |
| `poly_degree` | the degree in `x` of the manifold fit |
| `s_lo`, `s_hi` | the scale window searched, in multiples of the manifold density |
| `n_scan` | points of the bracketing scan over the window |
| `n_iter` | bisection steps after the scan |
| `n_species` | the channel count (two) |
| `pressure_route` | `protocol` or `closed_form` |

NaN where the window holds no rising root (no mechanically stable state). Of several roots, a rising
one nearest the manifold wins.

    isobar_density(model, P, x, T, **same) -> n(x)

`s(x) n_manifold(x)`, every parameter as `isobar_scale`; raises when no point of the path has a state.

    binodal_on_isobar(model, P, T, x, *, manifold, pressure_unit, poly_degree, s_lo, s_hi, n_scan,
                      n_iter, min_path_points, min_tie_gap, central_lo_max, central_width_max,
                      n_species, pressure_route) -> (x_lo, x_hi, max|s - 1|)

| parameter | meaning |
|---|---|
| `model`, `P`, `T`, `x`, `manifold`, `pressure_unit`, `poly_degree`, `s_lo`, `s_hi`, `n_scan`, `n_iter`, `n_species`, `pressure_route` | as `isobar_scale` |
| `min_path_points` | solved points below which the isobar gives no binodal |
| `min_tie_gap` | the narrowest composition gap a lower-hull facet must span to count as a tie |
| `central_lo_max` | a central tie's left end lies below this fraction |
| `central_width_max` | a central tie is narrower than this |

The widest central tie wins, else the widest tie; `(nan, nan, tilt)` when there is none.

`tangent_free_energy(model, x, n, T, P, *, pressure_unit, n_species)` is `g = (f + P) / n` along a
solved isobar.

## The declared block

Manifolds (`phase_diagram`, `stability_map`, `dome`):

| key | meaning |
|---|---|
| `eos_csvs` | `{pressure: relative path}`, read from the system's tracked tables `experiments/<system>/eos/` first, then from its raw root |
| `eos_columns` | `x_column`, `T_column`, `n_columns` (alternatives, the first present wins), `status_column`, `status_ok` |
| `manifold_fit` | `column` (blend two temperature columns, then fit; holes refused; needs `x_channel = 1`) or `per_row` (fit each temperature row on its measured nodes) |
| `manifold_min_nodes` | `per_row`: the measured compositions a row needs |
| `poly_degree`, `pressure_unit` | as above |
| `isobar` | `{"s_lo", "s_hi", "n_scan", "n_iter", "pressure_route"}` |

`phase_diagram` also reads:

| key | meaning |
|---|---|
| `x_grid`, `iso_x_grid` | the path's fractions, and the coarser grid `iso_s` is reported on |
| `binodal_route` | `convex_hull` (the only route this stage runs): the lower-hull facet that brackets `x_bar`. Any other value, `mu_roots` included, is refused by name before anything is written; `mu_roots` is the one-field read-off (`one_field["readoff"]`) |
| `x_bar` | the lever anchor, a fraction or `"auto"` (the most unstable interior point) |
| `min_path_points`, `min_tie_gap` | as `binodal_on_isobar` |
| `tc_method` | `dome_apex`, `ising_fit` or `bracket` |
| `apex` | `dome_apex`: `{"central_lo_max", "tail_fraction", "min_tail_points", "max_overshoot_steps"}`: the squared width `w^2 = a (T_c - T)` is fitted over the dome's tail, and a `T_c` beyond `max_overshoot_steps` grid steps above the last two-phase row is refused |
| `tc_kwargs` | the keywords of `ising_fit` or `bracket` |

`dome` and `tc` read `dome = {"pressures", "T_grid", "anchor_pressures", "x_grid", "close_after"}`
(a scan over `pressures`, each using the manifold of the nearest anchor pressure, stopping a
pressure after `close_after` single-phase rows), plus `isobar`, `pressure_unit`, `poly_degree`,
`min_path_points`, `min_tie_gap`, `central_lo_max`, `central_width_max`, `tc_method` and `apex`.

`stability_map` reads `stability_map = {"x_points": (lo, hi, count), "min_points", "hessian_dtype"}`
(`float32` or `float64`) and the manifold keys.

`one_field_phase_diagram` reads `one_field`:

| key | `mu_roots` | `bulk_minima` | meaning |
|---|---|---|---|
| `readoff` | yes | yes | the roots of the uniform `mu`, or a closed form's minima and the sign of `f''` |
| `T_grid` | yes | yes | the stage's grid |
| `dtype` | yes | yes | `float32` or `float64` |
| `phi_grid` | yes | yes | `{"lo", "hi", "n"}` |
| `phi_clip`, `min_width`, `bracket_eps` | yes | | the clamp of `phi`, the narrowest gap, the root bracket |
| `mu_w0`, `tc_w0` | yes | | `{"r_max", "n_points"}`: the quadrature of `W(0)` in `mu` and in `T_c` |
| `ising` | yes | | `{"T_max", "beta", "B0", "Tc0_floor", "Tc0_offset", "B_bounds", "Tc_bounds_offset", "maxfev"}`, the fit `delta(T) = (B/2)(1 - T/T_c)^beta`, or `None` |
| `edge_trim`, `centre` | | yes | grid points dropped at each end of `f''`; the mirror point |

## Other functions

`aipf.diagnose` also exports `binodal`, `binodal_convex_hull`, `binodal_mu_roots`, `spinodal`,
`critical_temperature`, `dome_apex`, `dome_tail` (`thermo`), `growth_rate`, `most_unstable_k`,
`structure_factor`, `finite_difference_rate`, `rollout_drift` (`dynamics`) and
`nonlocal_kernel_kappa_eff` (`kappa`).
