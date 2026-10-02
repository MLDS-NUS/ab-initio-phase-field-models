"""The two-way loss-name map: canonical paper names to each checkpoint layout's symbols.

Canonical: ``L_dyn``, ``L_M``, ``L_S``, ``L_bulk``, ``L_P``, ``L_conv``, ``L_Gamma``; ``L_W`` is an extra.
Two layouts, ``A`` (Lightning hparams) and ``B`` (k-modes), get separate maps per axis, never one merged
map: the checkpoint key ``lambda_S`` is ``L_S`` in layout A and ``L_bulk`` in layout B. The caller picks the
layout."""
from __future__ import annotations

from collections.abc import Mapping

#: The paper's seven canonical loss names.
CANONICAL_LOSSES: tuple[str, ...] = (
    "L_dyn", "L_M", "L_S", "L_bulk", "L_P", "L_conv", "L_Gamma",
)

#: Non-canonical, system-specific extras this package can still express.
NONCANONICAL_EXTRAS: tuple[str, ...] = ("L_W",)

_ALL_NAMES = frozenset(CANONICAL_LOSSES) | frozenset(NONCANONICAL_EXTRAS)


class LossNameMap:
    """A validated two-way ``{canonical name: code symbol}`` map for one checkpoint layout.

    Partial coverage is allowed; two canonical names sharing one code symbol is refused."""

    def __init__(self, canonical_to_code: Mapping[str, str], *, label: str):
        for name in canonical_to_code:
            if name not in _ALL_NAMES:
                raise ValueError(
                    f"{name!r} is not a canonical loss name or a declared "
                    f"non-canonical extra; known names are "
                    f"{sorted(_ALL_NAMES)}")
        self._label = str(label)
        self._canonical_to_code = dict(canonical_to_code)
        self._code_to_canonical: dict[str, str] = {}
        for canonical, code in self._canonical_to_code.items():
            if code in self._code_to_canonical:
                raise ValueError(
                    f"code symbol {code!r} is claimed by both "
                    f"{self._code_to_canonical[code]!r} and {canonical!r} "
                    f"in map {self._label!r}: a code symbol may name only "
                    f"one canonical loss")
            self._code_to_canonical[code] = canonical

    @property
    def label(self) -> str:
        return self._label

    @property
    def canonical_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._canonical_to_code))

    @property
    def code_symbols(self) -> tuple[str, ...]:
        return tuple(sorted(self._code_to_canonical))

    def to_code(self, canonical: str) -> str:
        """The code symbol for ``canonical`` in this layout, or raise."""
        try:
            return self._canonical_to_code[canonical]
        except KeyError:
            raise KeyError(
                f"{canonical!r} has no code symbol in map {self._label!r}; "
                f"declared canonical names: {self.canonical_names}") from None

    def to_canonical(self, code: str) -> str:
        """The canonical name for ``code`` in this layout, or raise."""
        try:
            return self._code_to_canonical[code]
        except KeyError:
            raise KeyError(
                f"code symbol {code!r} is not mapped to a canonical name in "
                f"map {self._label!r}; declared code symbols: "
                f"{self.code_symbols}") from None


# Checkpoint weight-key maps (saved ``hparams`` keys).

# Layout A. No ``L_dyn`` entry: the drift weight is hardcoded at 1
# (``aipf.train.ckpt_compat.IMPLICIT_WEIGHTS_A``).
WEIGHT_MAP_A = LossNameMap({
    "L_M": "lambda_M",
    "L_S": "lambda_S",
    "L_bulk": "lambda_bulk",
    "L_P": "lambda_P",
    "L_conv": "lambda_conv",
    "L_Gamma": "lambda_gamma",
    "L_W": "lambda_wpsd",
}, label="A-weights")

# Layout B: three loss terms only; ``L_S``, ``L_P``, ``L_conv``, ``L_Gamma``, ``L_W`` are absent by
# measurement.
WEIGHT_MAP_B = LossNameMap({
    "L_dyn": "lambda_drift",
    "L_M": "lambda_M",
    "L_bulk": "lambda_S",  # the trap: this layout's "S" names the BULK anchor
}, label="B-weights")


# Training-log-key maps: a separate axis from the weight keys, never bridged by string similarity.

LOG_MAP_A = LossNameMap({
    "L_dyn": "train/drift",
    "L_M": "train/anchor",
    "L_S": "train/static",
    "L_bulk": "train/bulk",
    "L_P": "train/pressure",
    "L_conv": "train/conv",
    "L_Gamma": "train/L_gamma",
    "L_W": "train/wpsd",
}, label="A-log")

LOG_MAP_B = LossNameMap({
    "L_dyn": "train/drift",
    "L_M": "train/anchor_M",
    "L_bulk": "train/anchor_S",
}, label="B-log")
