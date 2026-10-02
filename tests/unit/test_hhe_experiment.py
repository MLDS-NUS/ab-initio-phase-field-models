"""The published checkpoint, pinned by its digest so a silent substitution cannot pass.

The training run behind it left seventeen checkpoint files in a single directory: a
best-validation file, a link to the last one written, and a periodic file every two epochs.
They share their `hyper_parameters` EXACTLY and hold different weights -- the two nearest
neighbours differ in 39 of 66 `state_dict` tensors -- so a neighbouring file reproduces the
configuration and not the numbers, and every configuration check in this suite would stay green
while the model changed underneath it. The published model is the best-validation file; the
repository tracks it at ``data/hhe/ckpt/published/final.ckpt`` and the System declares it by its
md5 alone.

Two claims, kept apart on purpose, because they fail for different reasons:

* the System DECLARES that digest -- checked from the declaration;
* the file ON DISK still carries those bytes -- checked wherever the checkout carries the tracked
  copy (a checkout may leave the published checkpoints out, and then this skips, naming it).
"""
from __future__ import annotations

import hashlib
import pathlib
import re

import pytest

from aipf.system import Checkpoint, load

import declared_roots

#: A checkpoint path relative to a root, for the declaration's own refusals below.
PUBLISHED_PATH = "runs/published/final.ckpt"

#: Digest of the tracked published model's bytes, measured with `md5sum`.
PUBLISHED_MD5 = "ab44d9883f2a3dd27e43cd1c783907a9"

#: Directories that hold CODE.
CODE_DIRS = ("src", "tests", "experiments")

#: Checkpoint filenames of the `epoch=<e>-step=<s>.ckpt` shape that a file
#: under `CODE_DIRS` may name.
#: Empty, and an entry here is a reviewed decision rather than an oversight:
#: the list being empty is the claim the scan exists to back.
ALLOWED_OTHER_CHECKPOINTS: tuple[str, ...] = ()

_REPO = pathlib.Path(__file__).resolve().parents[2]

_CKPT_LITERAL = re.compile(r"epoch=\d+-step=\d+\.ckpt")


def _hhe():
    return load("hhe")


def _md5(path: pathlib.Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# The declaration
# ---------------------------------------------------------------------------

def test_the_system_declares_the_published_model_checkpoint():
    """Repointing the System at another file fails here, not silently."""
    s = _hhe()
    assert s.checkpoint is not None, (
        "the system declares no checkpoint: its published numbers then come "
        "from whichever file a caller happens to pick")
    assert s.checkpoint.path is None     # the tracked copy: declared by its digest alone
    assert s.checkpoint.md5 == PUBLISHED_MD5


def test_the_checkpoint_is_the_tracked_copy():
    """Declared by digest alone, so it resolves to the repository's tracked file."""
    from aipf.paths import published_checkpoint

    s = _hhe()
    assert s.checkpoint_file() == published_checkpoint("hhe")


def test_a_raw_root_override_does_not_move_the_tracked_checkpoint(monkeypatch, tmp_path):
    s = _hhe()
    before = s.checkpoint_file()
    monkeypatch.setenv("AIPF_RAW_HHE", str(tmp_path))
    assert s.checkpoint_file() == before


# ---------------------------------------------------------------------------
# The bytes on disk
# ---------------------------------------------------------------------------

def test_the_file_on_disk_still_carries_the_declared_digest():
    """Swapping the file's CONTENTS fails here. A checkout that leaves the published checkpoints out
    skips, naming the file."""
    declared_roots.published_or_skip("hhe")
    s = _hhe()
    path = s.verify_checkpoint()
    assert _md5(path) == PUBLISHED_MD5


def test_a_missing_file_under_a_present_root_is_refused_not_skipped(tmp_path):
    """The guard's whole point, exercised without touching the real tree.

    Matched on the refusal's own wording, not on the filename. `open` raises
    the same exception type and puts the same filename in its message, so a
    filename match cannot tell "this code refused" from "this code dropped
    the check and the file system raised on its way past".
    """
    ckpt = Checkpoint(path=PUBLISHED_PATH, md5=PUBLISHED_MD5)
    with pytest.raises(FileNotFoundError, match="missing or renamed") as excinfo:
        ckpt.verify_under(tmp_path)
    assert str(tmp_path / PUBLISHED_PATH) in str(excinfo.value)


def test_a_present_file_with_other_bytes_is_refused_and_both_digests_named(tmp_path):
    target = tmp_path / PUBLISHED_PATH
    target.parent.mkdir(parents=True)
    target.write_bytes(b"not the published model")
    with pytest.raises(ValueError) as excinfo:
        Checkpoint(path=PUBLISHED_PATH, md5=PUBLISHED_MD5).verify_under(tmp_path)
    message = str(excinfo.value)
    assert PUBLISHED_MD5 in message
    assert _md5(target) in message


def test_the_right_bytes_under_any_root_are_accepted(tmp_path):
    body = b"pretend this is a checkpoint"
    target = tmp_path / PUBLISHED_PATH
    target.parent.mkdir(parents=True)
    target.write_bytes(body)
    ckpt = Checkpoint(path=PUBLISHED_PATH, md5=hashlib.md5(body).hexdigest())
    assert ckpt.verify_under(tmp_path) == target


# ---------------------------------------------------------------------------
# One spelling, repository wide
# ---------------------------------------------------------------------------

def test_no_code_file_names_a_different_checkpoint_of_this_shape():
    """The declaration is the only opinion the code holds.

    A second spelling of the published model's file is a second thing to repoint, and
    repointing it would leave this file's own assertions green. The published model is the
    tracked ``final.ckpt``, so no `epoch=<e>-step=<s>.ckpt` literal is left under the code
    directories.
    """
    scanned, hits = 0, []
    for directory in CODE_DIRS:
        for path in sorted((_REPO / directory).rglob("*")):
            if not path.is_file() or path.suffix == ".pyc":
                continue
            scanned += 1
            text = path.read_text(encoding="utf-8", errors="replace")
            for match in _CKPT_LITERAL.finditer(text):
                found = match.group(0)
                if found not in ALLOWED_OTHER_CHECKPOINTS:
                    line = text[:match.start()].count("\n") + 1
                    hits.append(f"{path}:{line}: {found}")
    assert scanned > 0, f"scanned no files under {_REPO}"
    assert not hits, hits


# ---------------------------------------------------------------------------
# What the declaration itself refuses
# ---------------------------------------------------------------------------

def test_an_absolute_checkpoint_path_is_refused():
    with pytest.raises(ValueError, match="relative"):
        Checkpoint(path="/somewhere/" + PUBLISHED_PATH, md5=PUBLISHED_MD5)


def test_an_empty_checkpoint_path_is_refused():
    with pytest.raises(ValueError, match="empty"):
        Checkpoint(path="", md5=PUBLISHED_MD5)


@pytest.mark.parametrize("digest", [
    PUBLISHED_MD5.upper(),          # compared against `md5sum`, which is lower
    PUBLISHED_MD5[:31],             # a digest one character short
    PUBLISHED_MD5 + "0",            # and one too long
    "53c32fd018594e00f65832521fbccz28",   # not hexadecimal
    "",
])
def test_a_digest_that_is_not_thirty_two_lowercase_hex_digits_is_refused(digest):
    with pytest.raises(ValueError, match="md5"):
        Checkpoint(path=PUBLISHED_PATH, md5=digest)

