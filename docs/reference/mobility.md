# Mobility

`aipf.mobility.Mobility` is the Onsager mobility `M(rho, T)` every functional carries. It is
symmetric positive semi-definite for any parameter values:

    M(rho, T) = sqrt(g(rho)) M_tilde(rho[, T]) sqrt(g(rho)) * a(T) a(T)^T

with `M_tilde = L L^T` (a Cholesky factor with a softplus diagonal and free off-diagonal entries),
`g` the degeneracy prefactor and `a = sqrt(c(T))` the temperature factor, applied per channel.

## Declaring one

    Mobility(form, T_form, kwargs)

`form` is the declared name, mapped onto the class's `shape` axis (`MOBILITY_SHAPES`):

| `form` | `shape` | `M_tilde` |
|---|---|---|
| `mlp_scaled`, `mlp` | `mlp_rho` | an MLP of the density, two hidden layers |
| `mlp_rho_t` | `mlp_rho_t` | an MLP of the density and `T`; `T_form` must be `none` |
| `constant`, `gamma` | `constant` | a trained constant matrix, from `shape_init` |
| `fixed` | `fixed` | a constant matrix, not trained |
| `lattice_scalar` | `lattice_scalar` | the one-field form below |

`T_form` is `none` (`c = 1`) or `arrhenius`:
`c(T) = exp(-E (1/(kB T) - 1/(kB T_ref)))`, one barrier `E = softplus(raw)` per channel, in the
system's energy unit, with `kB` from `constants["kB"]`.

## kwargs

| declared key | constructor argument | admissible | needed by |
|---|---|---|---|
| `mobility_prefactor` | `prefactor` | `mole_fraction` (`g_i = rho_i (1 - rho_i)`), `partial_density` (`g_i = rho_i`) | every shape |
| `mobility_input_ref` | `input_ref` | one float per channel, or `None` | the MLP shapes, which read `rho / input_ref - 1` (`None`: `rho` itself); must be declared for them and is refused for the others |
| `mobility_activation_energy_init` | `activation_energy_init` | one raw barrier per channel (`softplus^-1` of the energy) | `arrhenius` |
| `arrhenius_shared_Ea` | | must be `False`: each channel has its own barrier | |
| `shape_init` | `shape_init` | `n(n+1)/2` raw Cholesky entries, or the raw `log gamma` for `lattice_scalar` | `fixed`, `constant`, `lattice_scalar` |
| `rho_eps` | `rho_eps` | float; else the functional's `rho_eps` | `lattice_scalar` only |
| any other key | its own name | as the constructor takes it | |

Read from the functional's kwargs: `h_m` (the MLP width, `hidden`), `activation` (`gelu`, `tanh`,
`silu`, `softplus`) and `T_ref` (the Arrhenius reference temperature, `t_ref`). `n_species` comes
from the system.

## lattice_scalar

One field, `prefactor = "mole_fraction"`:

    M = r (1 - r) exp(log_gamma) c(T),   r = clamp(rho, rho_eps, 1 - rho_eps)

with `c(T) = exp(-softplus(Ea_raw) (1/(kB T) - 1/(kB T_ref)))` under `arrhenius`. Two scalar
parameters.

## Functions

| function | returns |
|---|---|
| `mobility_kwargs(system, *, functional_kwargs=None, **overrides)` | the `Mobility` constructor arguments the system declares; overrides use the class's own names, and an unknown one is refused |
| `build(system, **overrides)` | a standalone `Mobility`, with `build_overrides` recorded |
| `mobility_prefactor(rho, form)` | the vector `g(rho)` |
| `cholesky_raw_to_matrix(raw, n_species)` | `L L^T` from the raw lower triangle, row-major |

The functionals build their own mobility from the same arguments, prefixed `mobility_` in their
constructors ([functional.md](functional.md)).

## Example

The Fe-B declaration, two channels, an MLP of the scaled density with one Arrhenius barrier per
species:

    Mobility(form="mlp_scaled", T_form="arrhenius", kwargs=dict(
        arrhenius_shared_Ea=False,
        mobility_prefactor="partial_density",
        mobility_input_ref=(0.05, 0.05),
        mobility_activation_energy_init=(math.log(math.expm1(0.38)), math.log(math.expm1(0.50)))))
