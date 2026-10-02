# Functionals

`aipf.functional` holds the free-energy ladder: one protocol, one registry, and the rungs that
implement it. A system names its rung in `System.functional`; `aipf.functional.build.build(system)`
turns that declaration into a model.

## The protocol

Every rung is a `torch.nn.Module` with the four methods of `aipf.functional.base.FreeEnergyModel`:

| method | in | out |
|---|---|---|
| `forward(rho_hat, boxes, T)` | `rho_hat` `(B, n_species, Gx, Gy, Gz//2+1)`, `boxes` `(B, 3)`, `T` `(B,)` | `d rho_hat / dt`, the Model B right-hand side `div(M grad mu)` in k-space; the `k = 0` mode is exactly zero |
| `chemical_potential(rho, boxes, T)` | real-space `rho` `(B, n_species, Gx, Gy, Gz)` | `mu`, same shape |
| `bulk_free_energy_density(rho, T)` | real-space `rho` | the pointwise local free energy, without gradient or kernel terms |
| `mobility(rho, T)` | real-space `rho` | `M(rho, T)`, `(B, n, n, Gx, Gy, Gz)` |

`MODEL_REGISTRY` maps a name to a class and checks the four methods at registration. Registered:
`landau`, `fh`, `square_gradient`, `nonlocal_kernel`, `neural_operator`.

## Declaring one

    Functional(form, local, kernel, kwargs)

| `form` | rung | `local` | `kernel` | built by `build` |
|---|---|---|---|---|
| `landau` | closed basis expansions (`Landau`, `FloryHuggins`) | | `None` | no: refused |
| `square_gradient` | local free energy plus `(1/2) kappa (grad rho)^2` | `landau`, `flory_huggins` (`quadratic_quartic` exists, untranslated) | `None` | yes |
| `nonlocal_kernel` | local free energy plus a pair kernel `(1/2) rho W * rho` | `mlp`, `icnn`, `taylor` | `radial_mlp` | yes |
| `neural_operator` | an equivariant operator, implemented and equivariance-tested, not trained | | `None` | no: refused |

`kwargs` are the constructor arguments in the declaration's spelling. The builder never uses a
constructor default: a parameter neither declared nor supplied by the system is refused, naming the
spelling to declare it under.

    aipf.functional.build.build(system, variant=None, **overrides)
    aipf.functional.build.rung_kwargs(system, variant=None, **overrides)

`rung_kwargs` returns the constructor arguments; `build` constructs the model and records the
overrides as `model.build_overrides`. An override is spelled the declaration's way for the renamed
keys below, else by its constructor name; an argument the constructor does not take is refused.

## nonlocal_kernel

`F = int [ f_loc(rho, kB T) + (1/2) rho^T (W * rho) ] dr`, with
`f_loc = f_id + f_exc` and a radial pair kernel `W_ab(r)`, one MLP per unordered species pair, times a
quintic envelope that vanishes at `R_cut`.

Supplied by the system, not declared: `n_species`, `kB` (`constants["kB"]`), `u_form`
(`Functional.local`), `mobility_shape` and `mobility_t_form` (from `System.mobility`).

Declared spellings that differ from the constructor's:

| declared | constructor | conversion |
|---|---|---|
| `h_w` | `kernel_hidden` | |
| `g_form` | `g_exc_form` | |
| `fexc_T_ref` | `kBT_ref` | multiplied by `kB` |
| `activation` | `local_activation`, `kernel_activation` (and the mobility's) | one value for all nets |
| `h_m`, `T_ref` | the mobility's `hidden`, `t_ref` | [mobility.md](mobility.md) |
| `tbasis_ortho` | | a switch: `True` needs `tbasis_ortho_window` and `tbasis_ortho_points`; `False` drops them |
| `disable_u`, `disable_g` | | must be `False` |

Constructor arguments and admissible values:

| argument | admissible | meaning |
|---|---|---|
| `grid` | three positive ints | the real-space grid |
| `rho_ref` | one float per channel | the excess nets' reference density |
| `R_cut` | float | the kernel's range |
| `h_g` | int | width of the excess-entropy net |
| `h_u` | int | width of the energy net; required by the split form with `u_form` `icnn` or `mlp` |
| `ideal_form` | `gas`, `lattice` | `kBT sum rho ln rho - rho`, or `kBT sum rho ln rho + (1 - rho) ln(1 - rho)` |
| `nyquist_mask` | bool | the gradient and divergence zero the Nyquist wavenumber |
| `f_exc_form` | `split`, `joint` | `u(rho) + kBT g_exc(rho) [+ T-basis heads]`, or one net `u_joint(rho, kBT / kBT_ref)` |
| `u_form` (`local`) | `mlp`, `icnn`, `taylor` | the energy net |
| `u_degree`, `u_parity`, `u_variable` | int; `full`, `even`; `ratio`, `difference` | the Taylor energy's degree, parity and variable; required with `taylor` |
| `gauge_fix` | bool | subtract the energy's affine part at `rho_ref` |
| `g_exc_form` (`g_form`) | `icnn`, `mlp` | the excess-entropy net |
| `g_symmetry` | `none`, `mirror` | `mirror` averages `g_exc(rho)` and `g_exc(1 - rho)`; lattice ideal term, split form, unscaled input only; required with `ideal_form="lattice"` |
| `icnn_output_bias` | bool | required when an ICNN is built |
| `kernel_argument` | `density`, `difference` | the kernel convolves `rho` or `rho - rho_ref`; required with `ideal_form="lattice"` |
| `enable_TlnT`, `enable_T2` | bool | the `kBT ln(kBT / kBT_ref)` and `kBT (kBT / kBT_ref)` heads; need `h_g_hat`, `h_g_tilde` |
| `h_g_hat`, `h_g_tilde` | int or `None` | the heads' widths |
| `tbasis_ortho_window`, `tbasis_ortho_points` | two energies, int | the window and points the heads are orthogonalised over |
| `h_joint`, `joint_depth` | int | the joint net's width and hidden layers; required with `joint` |
| `kBT_ref` (`fexc_T_ref`) | float | required by the joint form, the T-basis heads and their orthogonalisation |
| `local_input_scale` | bool | the excess nets read `rho / rho_ref - 1` instead of `rho` |
| `local_activation`, `kernel_activation` | `gelu`, `tanh`, `silu`, `softplus` | |
| `rho_eps` | float | the floor under the ideal term |
| `kernel_hidden` (`h_w`) | int | width of each radial net |
| `kernel_tail_sigma` | float or `None` | an optional tail `exp(-r^2 / (4 sigma^2))` on the envelope |
| `kernel_evaluator` | `None`, `"lattice_sum"` | `None` tabulates `W(k)` analytically from the radial transform; `lattice_sum` sums `W` over the grid's nearest-image distances |
| `kernel_n_quad`, `kernel_n_k_table`, `kernel_k_table_max` | int, int, float | the analytic evaluator's radial quadrature and k table; required with `kernel_evaluator=None` |

The joint form refuses `g_exc_form="mlp"` and the T-basis heads. Construction order (local form,
kernel, mobility, T-basis heads) fixes the random stream, so a model built under a given seed is the
same model every time.

Methods beyond the protocol: `uniform_free_energy(rho, T, w0)`, the free energy of uniform states;
`kernel.w_hat(k)`, `kernel.w_hat_zero_radial(r_max, n_points)` (`W(0)` by quadrature) and
`kernel.kappa_eff()`, the `k^2` coefficient of `W(k) = W(0) + kappa_eff k^2 + O(k^4)`, that is
`-(2 pi / 3) int r^4 W(r) dr` (the `kappa` stage of [diagnose.md](diagnose.md)).

## square_gradient

`F = int f_loc(rho, T) + (1/2) grad(rho)^T kappa grad(rho)`, one field for the closed local forms.

| `local` | `f_loc` | declared bulk constants |
|---|---|---|
| `flory_huggins` | `w rho (1 - rho) + T [rho ln rho + (1 - rho) ln(1 - rho)]` | `w` |
| `landau` | `(1/8) a0 (T - T_c) psi^2 + (b/16) psi^4`, `psi = 2 rho - 1` | `a0`, `b`, `T_c` |
| `quadratic_quartic` | `(1/2) rho^T A rho + (1/4) sum b_i rho_i^4`, `kappa` an SPD matrix, constant mobility | none; not translated by `build` |

Declared keys: `grid`, `nyquist_mask`, `kappa` (the starting `kappa`, constructor `kappa_init`),
`rho_eps` (the fraction clamp), the bulk constants, `gamma_fixed` (must be `None`: the mobility scale
is trained) and the mobility-owned `T_ref`. A constant a form does not read is refused. The mobility
must be one of the net-free shapes `fixed`, `constant`, `lattice_scalar`. The closed forms expose
`bulk_free_energy_curve(phi, T)` and `Tc_curvature()`, which the one-field diagnosis reads.

## landau and fh (basis expansions)

`Landau(grid, n_species, a0, b, T_c, gamma, *, nyquist_mask)` and
`FloryHuggins(grid, n_species, gamma, *, w=None, rho_ref=None, degree=4, rho_eps=1e-6, nyquist_mask)`,
registered as `landau` and `fh`. Every constant is a constructor argument; at `n_species > 1` the
Flory-Huggins enthalpy is a Redlich-Kister polynomial in `rho / rho_ref - 1`. The formulas are
module functions (`landau_f`, `landau_mu`, `regular_solution_f`, `ideal_mixing_f`, ...) that the
square-gradient forms call too.

## neural_operator

`NeuralOperator(grid, n_species, n_layers=3, hidden=8, *, nyquist_mask)`: each layer applies a radial
gain `c(|k|, T)`, a permutation-symmetric channel mix, a bias and `tanh`; `mu` is the stack's output
and the bulk free energy is zero. The mobility is `a I + b (ones - I)`, positive semi-definite by
construction.

## Adding a functional

A new rung is a class with the four protocol methods, registered with
`MODEL_REGISTRY.register(name, cls)`, its module listed in `build._FORM_MODULES`, its form added to
`aipf.system.FUNCTIONAL_FORMS`, and a translation in `build._TRANSLATIONS` that maps the declaration
onto the constructor. Until it has a translation, `build` refuses the form by name.
