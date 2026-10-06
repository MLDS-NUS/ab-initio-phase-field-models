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

## A model defined elsewhere

A model this package does not define is declared with a factory:

    Functional(form, local, kernel, kwargs, factory=make_model)

`build(system, **overrides)` calls `make_model(system, **overrides)` and returns what it gives back,
with the overrides recorded as `build_overrides`; `rung_kwargs` refuses such a functional, having
nothing to translate. `form`, `local` and `kernel` are then names only and are not checked against
the ladder. `kwargs` carries at least `grid` and `nyquist_mask`, which training and the rollouts read
off the declaration. The factory is left out of the functional's `repr` and equality. A variant
declares its own factory the same way, and its `mobility` may be `None`. A run's `MANIFEST.json`
records the factory as `model_factory: "module:qualname"` (a `functools.partial` is recorded as the
function it wraps).

`system.py` is executed outside `sys.modules`, so the factory and the classes it builds are imported
from an installed package, not defined in `system.py`. A checkpoint holds the model's `state_dict`
and no class: a reload builds the model from the declaration again and loads the weights strictly.

What the model must provide, and which part of the package reads it:

| what | read by | checked |
|---|---|---|
| a `torch.nn.Module` with at least one parameter | training, checkpoints | by `build` |
| `forward`, `chemical_potential`, `bulk_free_energy_density`, `mobility`, as in [the protocol](#the-protocol) | training, the solvers | by `build` |
| `_cache`, an `aipf.spectral.OpsCache` on the declared `grid`, and `ops`, the same object as `_cache.ops`, with the declared `nyquist_mask` | training (the operators per grid), the explicit solvers | by `build` |
| `kernel.w_hat(k)`: `(*k.shape, n, n)` from `|k|` alone, no grid or box argument | the semi-implicit scheme's frozen operator | when it runs |
| `f_local.f_pointwise(rho, kBT)` (`rho` `(P, n)`, `kBT` `(P,)`, out `(P,)`) and `f_local.mu_pointwise(rho, kBT)` (out `(P, n)`) | the semi-implicit scheme's frozen Hessian and its guards | when it runs |
| optionally `stabilizer_mobility(rho, T)` | the semi-implicit scheme's `M_s` ([rollout.md](rollout.md#the-scheme)) | when it runs |
| `mobility`, `bulk_free_energy_density`, `chemical_potential`; the field-wide curvature through `kernel.w_hat(k)`, else `curvature_at_wavevector(k)`, else zero sized by `n_species`; for the `W(0)` routes `radial` and `lattice`, `kernel.w_hat_zero_quadrature` and `kernel.evaluator` | the anchor tables, with `--anchors declared` ([training.md](training.md)) | when a batch is built; a missing one is refused by name |

The rollouts' guards act by replacing methods on the instance for the length of a rollout. In the
semi-implicit scheme `clamp_rho` replaces `f_local.mu_pointwise` and `mobility`, while `v_ext` and
`kbt_field` replace `f_local.mu_pointwise` only; in the explicit solvers `clamp_rho` replaces
`chemical_potential` and `mobility`. A
`forward` that reads `mu` through `self.chemical_potential`, the pointwise part of it through
`self.f_local.mu_pointwise`, and `M` through `self.mobility` sees every guard; a `forward` that
bypasses them does not, and must not be rolled out with those options.

`stabilizer_mobility` receives the field itself, unclamped; under `clamp_rho` the clamp reaches it only
if it reads `M` through `self.mobility`. A model with the hook takes no `m_stab`, while the
`spinodal`, `slab` and FDT drivers always pass the system's declared `Noise.m_stab`, so such a model
is rolled out by calling `aipf.rollout.imex.rollout_imex` directly. `aipf diagnose` reads parts of
this package's own rungs and refuses a functional built by a factory.

## Two dimensions

A model is two-dimensional when its declared `grid` has two entries, `(Gx, Gy)`. The number of axes
is read from that declaration (`model.ops.ndim`), never from a tensor's shape: a half spectrum
`(B, n, Gx, Gyr)` read as three axes would take the species axis for `Gx`. `aipf.spectral.make_ops`
builds `SpectralOps` for three axes and `SpectralOps2D` for two, and an `OpsCache` seeded with a
two-axis grid holds `SpectralOps2D` and reads two trailing axes on every lookup. In two dimensions the
half (`rfft`) axis is `y`:

| | 3D | 2D |
|---|---|---|
| `rho_hat` | `(B, n, Gx, Gy, Gz//2+1)` | `(B, n, Gx, Gy//2+1)` |
| `boxes` | `(B, 3)` | `(B, 2)`, `(Lx, Ly)` |
| `M(rho, T)` | `(B, n, n, Gx, Gy, Gz)` | `(B, n, n, Gx, Gy)` |
| persistent buffers | `NX`, `NY`, `NZ`, `MULT` `(1, 1, 1, 1, Gzr)` | `NX`, `NY`, `MULT` `(1, 1, 1, Gyr)` |
| `ops.ndim` | 3 (a class attribute, not state) | 2 |

`MULT` is the multiplicity along the half axis in both: `sum |full fft|^2 = sum MULT |rfft|^2`, with
the `k = 0` and, for an even length, the Nyquist entry counted once. The Nyquist masks of the odd-order
operators are the same rule on each axis.

What runs in two dimensions: `nonlocal_kernel`, a model built by a factory on a two-axis `grid`, the
explicit integrators and the semi-implicit scheme of [rollout.md](rollout.md#the-scheme), `hermitianize`
and the state projections, and `Field`. The pair kernel's `Ŵ(k)` is then the Hankel transform of the
same radial `W(r)`,

    Ŵ(k) = 2 pi int_0^R_cut r W(r) J0(k r) dr,

`AnalyticRadialTransform(..., dim=2)`, on the same `r_quad` and `k_table` buffers (`dim` is an
attribute, so the state dict is the 3D one's). A two-dimensional `nonlocal_kernel` builds it itself and
refuses an evaluator that does not declare `dim=2`.

What is three-dimensional only, and raises `NotImplementedError` naming itself when a two-dimensional
model or grid reaches it: the `landau`, `fh`, `square_gradient` and `neural_operator` rungs, the
`lattice_sum` evaluator, `W_hat(0)` (`w_hat_zero_radial`, `w_hat_zero_quadrature`) and `kappa_eff`,
the anchor tables, the convexity and Gamma-path penalties, the kernel hinge, `aipf diagnose` (and the
model isobars `aipf md` draws through it), the `slab`, `spinodal` and FDT drivers and their observables,
and the KDE deposit, whose positions are three-dimensional. A Lightning-hparams or k-modes checkpoint
drops its grid buffers on load, so the number of axes it was saved on (its `ops.NX` rank, 3 when not
saved) is compared with the model's first and a mismatch is refused; this package's own checkpoints
load strictly and differ in those buffers anyway.

The noise of a two-dimensional model needs one more number. Its variance is `2 kBT / (dV dt)`, and a
grid of `(Lx, Ly)` cells fixes only the area `dA` of a cell; `dV = dA * depth`, with `depth` declared
on every noisy solver call (`rollout_sde`, `step_sde_euler_maruyama`, `rollout_imex` with `noise`). It
says what the densities are: `depth = 1.0` for areal densities (per unit area), the reference cell's
`Lz` for volumetric densities averaged along z. There is no default; a noisy 2D call without `depth` is
refused, and a 3D call with one is refused too (its `dV` comes from its box). Deterministic 2D rollouts
read no `depth`. The stationary spectrum is then `S(k) = V <|rho_hat_k|^2> = kBT H(k)^-1` with
`V = Lx Ly depth`.

A two-dimensional model is an effective model, not the z average of a three-dimensional one. The
`k_z = 0` plane of 3D Model B dynamics is not closed: the nonlinear terms couple the `k_z != 0` modes
into it, so the z average of a 3D trajectory does not obey any functional of the z average alone. The
z-invariance check in `tests/unit/test_solve_2d.py` (a 2D rollout equals the 3D rollout of the same
field held constant along z, sliced in z) checks the solvers, which agree to float precision when the
3D field has no `k_z != 0` content to begin with; it says nothing about that closure. That agreement
holds under `nyquist_mask=True`, and under `nyquist_mask=False` when `Gx` is odd. With
`nyquist_mask=False` and an even `Gx`, the odd-order operators multiply the x-Nyquist line by
`i k_N`, which is not the transform of a real field there. The final inverse transform treats such a
line differently in the two layouts. In 3D the line lies on the `k_z = 0` plane and the last
(half-axis, z) inverse transform drops it. In 2D the half axis is y, so the line survives at every
interior `k_y` as a real `(-1)^x` pattern in x. A mask-off 2D model on an even `Gx` therefore differs
from its z-invariant 3D twin by such a term, of the size of the drift itself. 3D has the same
behaviour on its `k_z != 0` planes. The 2D operators are the plain `rfft2` ones and are left as they
are; `nyquist_mask=True` (or an odd `Gx`) is what makes the two layouts agree. The y-Nyquist
wavenumber, `-Gy/2` on the 3D full axis and `+Gy/2` on the 2D half axis, makes no difference. For
the same reason the
2D `W(r)` is not the 3D one: the z-projected kernel `int W3(sqrt(rho^2 + z^2)) dz` has as its Hankel
transform `Ŵ3(k)` at `k_z = 0`, and a 2D model learns its own `W` from 2D data.
