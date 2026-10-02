"""No tracked file says where the code came from: the public word gate.

The repository is read by people who never saw the trees it was assembled from, so a tracked file
names no earlier repository, no internal plan item, no machine, account or environment, no
training run, no session and no dated decision. Comments explain physics and usage; a checkpoint's
own ``hparams["..."]`` keys may be cited because the checkpoint is in the repository.

Text files are matched against :data:`WORDS`, case-insensitively, line by line. A binary file can
match a short word by chance inside compressed bytes, so it is matched against
:data:`BINARY_WORDS` only, long tokens that do not occur at random. This file is not scanned, as it
has to spell the words it forbids. A notebook is read as JSON: its cell sources and text outputs
against :data:`WORDS`, its whole text against :data:`BINARY_WORDS` (the embedded images are base64).

The binaries that carry text inside them, every tracked file under ``data/`` (the published
checkpoints: a pickle of settings beside the tensors) and every figure's ``figdata`` arrays and
images, are read as bytes and matched against :data:`BYTE_WORDS`. A zip archive (a checkpoint, an
``.npz``) is opened as well: every compressed member is matched decompressed, since a deflated string
is invisible in the raw bytes, and every string or object array in an ``.npy`` member (or file) is
matched as text, since numpy stores a ``str`` array as UTF-32, which no byte pattern meets. Without a
git index (an archive, a downloaded zip) the walk keeps to what git would track: under ``data/``
only the published checkpoints, never what the quick start writes beside them. A checkpoint's pickle record is matched against
:data:`PICKLE_WORDS` as well, which adds the short words that compressed bytes could produce by
chance but a pickle of settings cannot.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

#: What is scanned, relative to the repository root. Widening the gate is one edit here.
ROOTS: tuple[str, ...] = (
    "src", "experiments", "figures", "env", "tests", "docs",
    "README.md", "pyproject.toml", ".gitignore", "aipf.toml.example", "CITATION.cff",
)

#: Name -> pattern. Each names what it keeps out.
WORDS: dict[str, str] = {
    "an earlier repository": r"legacy|hhe-meso|hhe_meso|feb_meso|binary_lj",
    "a plan item": r"\btask ?[A-Z]?\d|\bphase ?\d|spec ?§|\(spec\s*$|\(spec (?:section|global)|\bspec section|§"
                   r"|(?-i:\b[A-G]\d[a-z]\b)|\bitem ?C\d\d\b|\bspec ?sec|\bsec1\d\b",
    "a migration date": r"(?<!date-released: )\b2026-\d\d-\d\d\b",
    "a decision record": r"\buser ruling\b",
    "this machine": r"\bnscc\b|asp2a|e0945231|(?<![.@\w])nus\b(?!\.edu)|this host|11004368|\bpbs10\d|\bq2@",
    "an environment name": r"torchenv|lammps_env|apfm",
    "a run name": r"champion|\bwave ?\d|\bwave_|pre_wave|wave\d|w35d4s4|lg1_s2|gammamlp|freeu|p30m50"
                  r"|_seed\d|epoch=\d+-step|\bprod_e\d+|\bbase_s\d",
    "an absolute path": r"/(?:home|scratch|tmp)/",
    "a session": r"claude|superpowers",
    "provenance prose": r"\bported from|verbatim \(provenance\)|\bthe tree\b|old script",
}

#: The long tokens of :data:`WORDS`, for bytes that are not text.
BINARY_WORDS: dict[str, str] = {
    "an earlier repository": r"legacy|hhe-meso|hhe_meso|feb_meso|binary_lj",
    "this machine": r"e0945231|asp2a|11004368",
    "an environment name": r"torchenv|lammps_env",
    "a run name": r"champion|w35d4s4|lg1_s2|gammamlp|p30m50|epoch=\d+-step",
    "an absolute path": r"/(?:home|scratch|tmp)/",
    "a session": r"claude|superpowers",
}

#: A figure's ``figdata`` files with these suffixes are arrays and images, not text: they are scanned as
#: bytes against :data:`BYTE_WORDS` and not as text against :data:`WORDS`.
FIGDATA_BINARY_SUFFIXES = (".npz", ".npy", ".pdf", ".png", ".pkl")

#: Where the byte gate looks: every tracked file under the first, the ``figdata`` binaries under the rest.
BYTE_ROOTS: tuple[str, ...] = ("data", "figures", "experiments")

#: A machine, an account, an absolute path or a run name, as bytes, five characters or longer so that
#: compressed bytes do not produce one by chance.
BYTE_WORDS: dict[str, bytes] = {
    "this machine": rb"e0945231|asp2a|11004368|/users/nus/",
    "an absolute path": rb"/(?:home|scratch|tmp)/",
    "a run name": rb"champion|w35d4s4|lg1_s2|gammamlp|p30m50|base_s\d|prod_e\d|_seed\d|wave ?\d|pre_wave|epoch=\d+-step",
    "an earlier repository": rb"legacy|hhe-meso|hhe_meso|feb_meso|binary_lj",
    "an environment name": rb"torchenv|lammps_env",
    "a session": rb"claude|superpowers",
}

#: The short words added for a checkpoint's pickle record, which is settings and not random bytes.
#: A key named ``seed`` would be a setting; a ``seed`` followed by a digit is a run name.
PICKLE_WORDS: dict[str, bytes] = {
    **BYTE_WORDS,
    "this machine, short": rb"nscc|scratch|zhongpc",
    "a run name, short": rb"seed\d|wave_|freeu",
    "an environment name, short": rb"apfm",
}

#: Directories never tracked (see .gitignore), skipped when there is no git index to ask.
_UNTRACKED_DIRS = {"__pycache__", ".pytest_cache", "out", ".ipynb_checkpoints"}

_SELF = Path(__file__).resolve()


def _compile(words: dict[str, str]) -> list[tuple[str, re.Pattern]]:
    return [(name, re.compile(pattern, re.IGNORECASE)) for name, pattern in words.items()]


def _tracked() -> list[Path]:
    """The tracked files under :data:`ROOTS`; a checkout without git (an archive) is walked instead."""
    try:
        out = subprocess.run(["git", "ls-files", "-z", "--", *ROOTS], cwd=REPO,
                             capture_output=True, check=True).stdout
        names = [n for n in out.decode().split("\0") if n]
        if names:
            return [REPO / n for n in names if (REPO / n).is_file()]
    except (OSError, subprocess.CalledProcessError):
        pass
    found = []
    for root in ROOTS:
        path = REPO / root
        if path.is_file():
            found.append(path)
            continue
        for directory, subdirs, files in os.walk(path):
            subdirs[:] = [d for d in subdirs if d not in _UNTRACKED_DIRS]
            found.extend(Path(directory) / f for f in files if not f.endswith((".pyc", ".pyo")))
    return found


def _scanned() -> list[Path]:
    def binary_figdata(p: Path) -> bool:
        return "figdata" in p.relative_to(REPO).parts and p.suffix.lower() in FIGDATA_BINARY_SUFFIXES
    return [p for p in _tracked() if p.resolve() != _SELF and not binary_figdata(p)]


def _hits_in_text(where: str, text: str, words) -> list[str]:
    hits = []
    for number, line in enumerate(text.splitlines(), 1):
        for name, pattern in words:
            for match in pattern.finditer(line):
                hits.append(f"{where}:{number}: {name}: {match.group(0)!r} in {line.strip()[:120]!r}")
    return hits


def _notebook_text(raw: str) -> str:
    """A notebook's cell sources and text outputs, one block per cell."""
    book = json.loads(raw)
    blocks = []
    for cell in book.get("cells", []):
        blocks.append("".join(cell.get("source", [])))
        for output in cell.get("outputs", []):
            blocks.append("".join(output.get("text", [])))
            blocks.append("".join((output.get("data") or {}).get("text/plain", [])))
    blocks.append(json.dumps(book.get("metadata", {})))
    return "\n".join(blocks)


def _hits(path: Path) -> list[str]:
    where = str(path.relative_to(REPO))
    data = path.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return _hits_in_text(where, data.decode("latin-1"), _compile(BINARY_WORDS))
    if path.suffix == ".ipynb":
        return (_hits_in_text(where, _notebook_text(text), _compile(WORDS))
                + _hits_in_text(where, text, _compile(BINARY_WORDS)))
    return _hits_in_text(where, text, _compile(WORDS))


def test_the_gate_scans_every_root():
    """An empty or mistyped root would report clean forever."""
    files = _scanned()
    for root in ROOTS:
        assert any(p == REPO / root or (REPO / root) in p.parents for p in files), \
            f"the gate scanned nothing under {root}"


def test_the_word_list_catches_what_it_names():
    """Each pattern fires on the shape it is for, and stays quiet on its near neighbours."""
    words = _compile(WORDS)

    def names(line: str) -> set[str]:
        return {name for name, pattern in words if pattern.search(line)}

    assert "a plan item" in names("see Task 5 and Phase 3, (spec §17.2), item C2c")
    assert "this machine" in names("the cluster's nus home") and "this machine" in names("on this host")
    assert "this machine" not in names("email: someone@u.nus.edu")
    assert "this machine" not in names("the minus sign") and "a plan item" not in names("module_from_spec(spec)")
    assert "a plan item" not in names("(spec or {})") and "a plan item" not in names("d2f/dx2")
    assert "provenance prose" in names("ported from the old code") and "provenance prose" not in names("exported from it")
    assert "a migration date" in names("by the ruling of 2026-09-22")
    assert "a migration date" not in names("date-released: 2026-10-01")
    assert "a run name" in names("wave20_gammaMLP_lg1_s2_seed2") and "a run name" not in names("wavevector")
    assert "a plan item" not in names("the phase diagram") and "a plan item" not in names("C1 potentials")
    assert "a plan item" in names("(task C12)") and "a plan item" in names("item C12 of the plan")
    assert "a plan item" not in names("the elastic constants C11 and C12")
    assert "a plan item" in names("(spec " + "sec17.1)") and "a plan item" in names("see sec" + "17")
    assert "a plan item" not in names("seconds") and "a plan item" not in names("t_sec10")
    assert "a run name" in names("M_table.pre_" + "wave15_clean.npz") and "a run name" in names("x_wave" + "15")
    assert "a run name" in names("x_wave" + "20_s1") and "a run name" in names("runs/x_wave" + "2")
    for physics in ("wavevector", "wavenumber", "a triangle wave", "the wave vector k", "waves"):
        assert "a run name" not in names(physics), physics
    assert "a run name" in names("runs/x_seed4/" + "epoch=6-" + "step=7.ckpt")
    assert "a run name" in names("prod_" + "e25_x_s1")
    assert "a run name" in names("fh/base_s1") and "a run name" not in names("database_size")
    assert "this machine" in names("-P 11004368") and "this machine" in names("q2@pbs101")
    assert "a session" in names("bin/docs/superpowers") and "an absolute path" in names("/tmp/x/y")


def test_no_tracked_file_names_where_the_code_came_from():
    hits = [hit for path in _scanned() for hit in _hits(path)]
    assert not hits, "\n".join(hits)


def _tracked_by_layout(rel: str) -> bool:
    """Without a git index: under ``data/`` only ``data/<system>/ckpt/published/**`` is tracked (see
    ``.gitignore``); the rest of the farm, ``diagnose/`` runs included, is the user's own output."""
    parts = Path(rel).parts
    return parts[0] != "data" or parts[2:4] == ("ckpt", "published")


def _byte_scanned() -> list[Path]:
    """Every tracked file under ``data/``, and every ``figdata`` binary; walked when there is no git index."""
    try:
        out = subprocess.run(["git", "ls-files", "-z", "--", *BYTE_ROOTS], cwd=REPO,
                             capture_output=True, check=True).stdout
        names = [n for n in out.decode().split("\0") if n]
    except (OSError, subprocess.CalledProcessError):
        names = []
    if not names:
        for root in BYTE_ROOTS:
            for directory, subdirs, files in os.walk(REPO / root):
                subdirs[:] = [d for d in subdirs if d not in _UNTRACKED_DIRS]
                names.extend(str((Path(directory) / f).relative_to(REPO)) for f in files)
        names = [n for n in names if _tracked_by_layout(n)]
    picked = []
    for name in names:
        parts = Path(name).parts
        if parts[0] == "data" or ("figdata" in parts
                                  and Path(name).suffix.lower() in FIGDATA_BINARY_SUFFIXES):
            if (REPO / name).is_file():
                picked.append(REPO / name)
    return picked


def _string_fields(dtype) -> bool:
    """Whether ``dtype`` holds text: a ``str``/``bytes``/object dtype, or a structured dtype with such a
    field (at any depth, sub-arrays included)."""
    if dtype.names:
        return any(_string_fields(dtype.fields[name][0]) for name in dtype.names)
    if dtype.subdtype is not None:
        return _string_fields(dtype.subdtype[0])
    return dtype.kind in "USO"


def _flat_strings(item, out: list) -> None:
    """Every string, bytes or object leaf of one array element (a record, a tuple, a nested list)."""
    if isinstance(item, bytes):
        out.append(item.decode("utf-8", "replace"))
    elif isinstance(item, str):
        out.append(item)
    elif isinstance(item, (tuple, list)):
        for part in item:
            _flat_strings(part, out)
    elif hasattr(item, "tolist") and not isinstance(item, (int, float, complex)):
        _flat_strings(item.tolist(), out)
    elif item is not None and not isinstance(item, (int, float, complex, bool)):
        out.append(str(item))


def _array_text(data: bytes) -> bytes:
    """The strings of an ``.npy`` string or object array, or of a structured array's string fields,
    one per line, as UTF-8; empty otherwise."""
    import io

    import numpy as np
    handle = io.BytesIO(data)
    try:
        version = np.lib.format.read_magic(handle)
        read = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                else np.lib.format.read_array_header_2_0)
        dtype = read(handle)[2]
    except ValueError:
        return b""
    if not _string_fields(dtype):
        return b""
    handle.seek(0)
    array = np.lib.format.read_array(handle, allow_pickle=dtype.hasobject)
    texts: list = []
    for item in array.ravel().tolist():
        _flat_strings(item, texts)
    return "\n".join(texts).encode("utf-8")


def _members(path: Path) -> list[tuple[str, bytes]]:
    """What the raw bytes of ``path`` may hide: a zip archive's compressed members, decompressed, and
    the strings of every ``.npy`` string or object array (a member, or the file itself)."""
    found = []
    if path.suffix.lower() == ".npy":
        found.append(("strings", _array_text(path.read_bytes())))
    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                is_npy = info.filename.endswith(".npy")
                if info.compress_type == zipfile.ZIP_STORED and not is_npy:
                    continue
                data = archive.read(info)
                if info.compress_type != zipfile.ZIP_STORED:
                    found.append((info.filename, data))
                if is_npy:
                    found.append((f"{info.filename}:strings", _array_text(data)))
    return [(name, data) for name, data in found if data]


def _byte_hits(where: str, data: bytes, words: dict[str, bytes]) -> list[str]:
    hits = []
    for name, pattern in words.items():
        for match in re.finditer(pattern, data, re.IGNORECASE):
            context = data[max(0, match.start() - 40):match.end() + 40]
            hits.append(f"{where}@{match.start()}: {name}: {match.group(0)!r} in {context!r}")
    return hits


def _pickle_records(path: Path) -> list[tuple[str, bytes]]:
    """A ``torch.save`` archive's pickle records (``<prefix>/data.pkl``), as bytes; none for other files."""
    if not zipfile.is_zipfile(path):
        return []
    with zipfile.ZipFile(path) as archive:
        return [(info.filename, archive.read(info)) for info in archive.infolist()
                if info.filename.endswith(".pkl")]


def test_the_byte_gate_reads_every_published_checkpoint_and_its_pickle():
    """An empty target list, or a checkpoint whose pickle is never opened, would report clean forever.

    A checkout may come without the published checkpoints; the checkpoint half then skips, naming
    them, after the figure data have been checked."""
    files = _byte_scanned()
    assert any("figdata" in p.parts for p in files), "the byte gate scanned no figdata binary"
    checkpoints = [p for p in files if p.suffix == ".ckpt"]
    if not checkpoints:
        import declared_roots
        for system in ("hhe", "feb", "lj"):
            declared_roots.published_or_skip(system)
    assert checkpoints and all(p.relative_to(REPO).parts[0] == "data" for p in checkpoints)
    for path in checkpoints:
        assert [name for name, _ in _pickle_records(path)], f"{path}: no pickle record found"


#: :data:`BINARY_WORDS` as byte patterns, for the self-test below.
BINARY_WORDS_BYTES: dict[str, bytes] = {k: v.encode() for k, v in BINARY_WORDS.items()}


def test_the_byte_words_catch_what_they_name():
    """Each byte pattern fires on the shapes the published checkpoints once carried, and not on settings."""
    def names(blob: bytes, words: dict[str, bytes]) -> set[str]:
        return {name for name, pattern in words.items() if re.search(pattern, blob, re.IGNORECASE)}

    # synthetic, and assembled from pieces, so that the gates do not read this file as naming them
    host = b"\x00/" + b"scratch/users/" + b"nus/someone/proj/runs/x_" + b"seed2/y"
    assert names(host, BYTE_WORDS) >= {"this machine", "an absolute path", "a run name"}
    assert "an absolute path" in names(b"X\x00/" + b"home/someone/x/y", BYTE_WORDS)
    assert "an absolute path" in names(b"X\x00/" + b"tmp/someone/x/y", BINARY_WORDS_BYTES)
    assert "a run name" in names(b"runs/x_" + b"seed4/", BYTE_WORDS)
    assert "a run name" in names(b"checkpoints/fh/base_" + b"s1/checkpoints", BYTE_WORDS)
    assert "an earlier repository" in names(b"binary_" + b"LJ_debug", BYTE_WORDS)
    assert "a run name, short" in names(b"lg9_" + b"seed7", PICKLE_WORDS)
    assert not names(b"\x8c\x04seed\x94K\x00 wavevector m_table fdt_800GPa/M_table.npz", PICKLE_WORDS)


@pytest.mark.parametrize("save", ["savez", "savez_compressed"])
@pytest.mark.parametrize("kind", ["str", "bytes", "object"])
def test_a_string_array_in_an_npz_is_read_as_text(tmp_path, save, kind):
    """A ``str`` array is UTF-32 in the ``.npy`` and a deflated one is not in the raw bytes at all;
    the member scan finds the path either way, compressed or stored."""
    import numpy as np

    path = tmp_path / "x.npz"
    secret = "/" + "scratch/" + "users/x"
    array = {"str": np.array([secret]), "bytes": np.array([secret.encode()]),
             "object": np.array([secret, 1], dtype=object)}[kind]
    getattr(np, save)(path, note=array)
    if kind == "str":
        assert not _byte_hits("x.npz", path.read_bytes(), BYTE_WORDS)
    members = _members(path)
    assert any(_byte_hits(f"x.npz:{name}", data, BYTE_WORDS) for name, data in members)


@pytest.mark.parametrize("save", ["save", "savez", "savez_compressed"])
@pytest.mark.parametrize("field", ["U64", "S64", "O", "(2,)U64"])
def test_a_string_field_of_a_structured_array_is_read_as_text(tmp_path, save, field):
    """A structured dtype hides its ``U``/``S``/object fields from a check on ``dtype.kind`` (it is
    ``V``); the member scan reads every string field, nested sub-arrays included."""
    import numpy as np

    secret = "/" + "scratch/" + "users/x"
    dtype = np.dtype([("T", "f8"), ("where", field)])
    value = (secret.encode() if field.startswith("S")
             else (secret, "ok") if field.startswith("(2,)") else secret)
    array = np.array([(1.0, value)], dtype=dtype)
    if save == "save":
        path = tmp_path / "x.npy"
        np.save(path, array, allow_pickle=True)
    else:
        path = tmp_path / "x.npz"
        getattr(np, save)(path, record=array)
    members = _members(path)
    assert any(_byte_hits(f"x:{name}", data, BYTE_WORDS) for name, data in members)
    plain = tmp_path / "plain.npy"
    np.save(plain, np.zeros(3, dtype=[("T", "f8"), ("n", "i4")]))
    assert _array_text(plain.read_bytes()) == b""


def test_without_a_git_index_the_walk_keeps_to_what_git_would_track():
    assert _tracked_by_layout("data/hhe/ckpt/published/final.ckpt")
    assert _tracked_by_layout("data/lj/ckpt/published/fh/final.ckpt")
    assert _tracked_by_layout("figures/fig4_feb/figdata/gamma_map.npz")
    for written in ("data/hhe/diagnose/ab44d9883f2a/MANIFEST.json", "data/lj/ckpt/some_run/final.ckpt",
                    "data/feb/manifest.json"):
        assert not _tracked_by_layout(written), written


def test_no_tracked_binary_names_a_machine_a_path_or_a_run():
    hits = []
    for path in _byte_scanned():
        where = str(path.relative_to(REPO))
        hits += _byte_hits(where, path.read_bytes(), BYTE_WORDS)
        for name, member in _members(path):
            hits += _byte_hits(f"{where}:{name}", member, BYTE_WORDS)
        for name, record in _pickle_records(path):
            if path.suffix == ".ckpt":
                hits += _byte_hits(f"{where}:{name}", record, PICKLE_WORDS)
    assert not hits, "\n".join(hits)
