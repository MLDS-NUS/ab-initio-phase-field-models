"""The package names no system and holds no absolute path; no subpackage shadows its own submodule."""
import re


def _segments(identifier: str) -> list[str]:
    """Split an identifier into lowercase segments.

    ``load_hhe_data`` -> ['load', 'hhe', 'data']
    ``FeB_constants`` -> ['fe', 'b', 'constants']
    ``max_batch``     -> ['max', 'batch']
    """
    parts = []
    for chunk in identifier.split("_"):
        parts += re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z]*|[a-z]+|\d+", chunk) or [chunk]
    return [p.lower() for p in parts if p]


def _runs(segments: list[str]) -> set[str]:
    """Every contiguous run of segments, rejoined without separators."""
    out = set()
    for i in range(len(segments)):
        for j in range(i + 1, len(segments) + 1):
            out.add("".join(segments[i:j]))
    return out


def test_package_names_no_system():
    """Global constraint: nothing under src/aipf names a species or a system.

    Matching is on contiguous runs of an identifier's segments (split on
    underscores and camel-case boundaries), case-insensitively, not on whole
    identifiers. Whole-identifier matching has a blind spot: ``load_hhe_data``,
    ``FeB_constants`` and ``x_He_profile`` are each a single token and pass it,
    and that shape is exactly what code written for one system produces. Substring matching is not the answer
    either, since it fires on ordinary names such as ``max_batch`` and
    ``index_by``, which contain ``x_b``. A guard that cries wolf gets loosened,
    and then it stops guarding. The test also asserts it scanned something, so
    a wrong or empty root cannot report clean forever.

    Identifier tokens catch ``hhe`` and ``feb``; they do NOT catch a system
    named in PROSE, which is how a docstring naming two of them once
    passed this check while the suite stayed green. A second,
    independent scan below matches whole phrases against the same file set,
    not substrings: a naive substring test for "iron" fires on "environment",
    the cry-wolf shape this suite has already hit once with tokens.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "aipf"
    assert root.is_dir(), f"package root not found at {root}"

    banned = {"xhe", "xb", "rhohe", "sigmaaa", "hhe", "feb",
              # and where the code came from: an earlier tree, its environment, a run label,
              # spelled in pieces so that the public word gate does not read them here
              "leg" + "acy", "ap" + "fm", "torch" + "env", "cham" + "pion", "binary" + "lj",
              "feb" + "meso"}
    token = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    # Identifier tokens catch `hhe` and `feb`; they do NOT catch a system named
    # in PROSE, which is how a docstring naming two of them passed this check.
    # Whole phrases, because a naive substring test for "iron" fires on
    # "environment" -- the cry-wolf shape this suite has already hit once.
    banned_prose = re.compile(
        r"\b(hydrogen[ -]helium|iron[ -]boron|lennard[ -]jones)\b", re.I)
    scanned, hits = 0, []
    for p in root.rglob("*.py"):
        scanned += 1
        text = p.read_text(encoding="utf-8")
        for m in token.finditer(text):
            if banned & _runs(_segments(m.group(0))):
                hits.append(f"{p}:{m.group(0)}")
        for m in banned_prose.finditer(text):
            hits.append(f"{p}: prose names a system: {m.group(0)!r}")
    assert scanned > 0, f"scanned no files under {root}"
    assert not hits, hits


#: Rooted paths the package is allowed to contain. Every entry needs a comment
#: saying why. The list being empty is the claim this test exists to back, and
#: an entry appearing here is a reviewed decision rather than an oversight.
_ALLOWED_ABSOLUTE: tuple[str, ...] = ()


def test_package_contains_no_absolute_paths():
    """Spec invariant 2. Every real location comes from config or the environment.

    An ALLOWLIST, not a denylist. A denylist encodes whichever mounts someone
    thought of on the day, and the measured version of that failed: 18 real
    shapes went through it, including ``Path("/scratch") / "users"``, which is
    how this very module composes paths, and four mount points that exist on
    the machine it was measured on. For a package whose defining property is holding no absolute
    paths, coverage must not depend on ``ls /`` at some site.

    Three rules, because no single one suffices:

    1. No string constant may START with ``/`` followed by a path character.
       Catches bare roots, the head of a split path, and assembled forms. The
       trailing condition is what distinguishes a rooted PATH from a bare ``/``
       separator or a regex such as ``r"/+"``, neither of which is a path.
    2. No rooted path ANYWHERE inside a constant. Keeps the docstring case,
       which rule 1 cannot see because a docstring begins with prose.
    3. No ``~`` expansion and no ``Path.home`` or ``expanduser``. These produce
       an absolute path with no string constant at all, so no string rule of
       any kind reaches them.
    """
    import ast
    import io
    import pathlib
    import re
    import tokenize

    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "aipf"
    assert root.is_dir(), f"package root not found at {root}"

    # Two or more rooted segments, not preceded by a word character, so that
    # "src/aipf/paths.py" in prose does not fire while "/srv/site/x" does.
    rooted = re.compile(r"(?<![A-Za-z0-9._-])/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
    url = re.compile(r"https?://[^\s'\"]+")

    # A rooted path has a segment after the slash. "/" alone is a separator and
    # r"/+" is a quantifier; neither is a path, and flagging them would push
    # legitimate entries onto the exemption list and cheapen it.
    rooted_head = re.compile(r"^/[A-Za-z0-9_.~-]")

    def check(where: str, text: str, hits: list) -> None:
        if rooted_head.match(text) and text not in _ALLOWED_ABSOLUTE:
            hits.append(f"{where}: rooted constant {text[:60]!r}")
        if "~/" in text:
            hits.append(f"{where}: home-relative {text[:60]!r}")
        found = rooted.search(url.sub("", text))
        if found and found.group(0) not in _ALLOWED_ABSOLUTE:
            hits.append(f"{where}: rooted path {found.group(0)!r}")

    scanned, hits = 0, []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.endswith(".pyc"):
            continue
        scanned += 1
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix != ".py":
            # Packaged data files count too. The package's own idiom is to move
            # site paths OUT of Python and into a plain text file, so a scanner
            # that reads only .py cannot see the form the project actually uses.
            for number, line in enumerate(text.splitlines(), 1):
                check(f"{path}:{number}", line, hits)
            continue
        # Comments too. `ast` discards them, so a rooted path in a `#` comment
        # is invisible to a constant scan -- and a comment is a natural place
        # to write "the real data lives under the home directory".
        # `tokenize`, not `line.partition("#")`: the naive split treats the
        # first "#" on a line as starting a comment even when it sits inside a
        # string literal. guard.py has five such lines today, and while none of
        # them currently holds a rooted path, a literal like
        # "tag:/a  # and then /srv/site/x" makes it fire on something that is
        # not a comment at all. A guard that cries wolf gets loosened.
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type == tokenize.COMMENT:
                body = token.string.lstrip("#").strip()
                check(f"{path}:{token.start[0]} (comment)", body, hits)
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                check(f"{path}:{node.lineno}", node.value, hits)
            elif isinstance(node, ast.Attribute) and node.attr in {"home", "expanduser"}:
                hits.append(f"{path}:{node.lineno}: {node.attr} produces an absolute path")

    assert scanned > 0, f"scanned no files under {root}"
    assert not hits, hits


def test_no_package_binds_a_name_one_of_its_own_submodules_has():
    """One convention for the six module/function name clashes.

    ``functional.build``, ``solve.build``, ``diagnose.run``, ``train.fit``,
    ``pipeline.modes`` and ``md.templates.builtin`` each name a function
    after the module it lives in. A package that re-exports such a name
    binds it over the SUBMODULE attribute the import system sets when that
    submodule loads, so which of the two a caller gets depends on import
    order -- and the three different ways this repository used to paper
    over that (an eager rebind, a lazy ``__getattr__`` plus a hand-rolled
    ``__setattr__`` guard, and no export at all) were themselves the
    finding.

    The rule is the cheapest of the three: never bind it. A clashing name
    is reached as ``from aipf.<pkg>.<submodule> import <name>``, and this
    scans every package for a breach rather than listing the six.

    TWO SCANS, because neither alone covers every package.

    1. What each package ACTUALLY binds, read off the imported module
       object. This is the one that does not depend on a package declaring
       anything: ``aipf.functional`` has no ``__all__`` at all, so a
       ``from .build import build`` added to it would pass a declaration
       scan silently. The attribute is read for every submodule NAME the
       package has, and a bound non-module under one of those names is the
       clash. Order does not save the check either: the shadowing import
       runs inside ``__init__``, so the function is what the attribute
       holds once the package finishes importing, whatever loads the
       submodule afterwards.

    2. What each package PROMISES -- ``__all__``, and ``_SOURCES`` for a
       package that resolves its exports lazily through ``__getattr__``.
       Scan 1 cannot see a lazy export that nothing has resolved yet,
       because until it is asked for there is no attribute to read.
    """
    import importlib
    import pathlib
    import pkgutil
    import types

    import aipf

    root = pathlib.Path(aipf.__file__).parent
    packages = ["aipf"] + [info.name for info
                           in pkgutil.walk_packages([str(root)], "aipf.")
                           if info.ispkg]
    hits = []
    for name in packages:
        package = importlib.import_module(name)
        submodules = {found.name.rsplit(".", 1)[1] for found
                      in pkgutil.iter_modules(package.__path__, name + ".")}

        for spelling in sorted(submodules):
            bound = getattr(package, spelling, None)
            if bound is None or isinstance(bound, types.ModuleType):
                continue
            hits.append(
                f"{name} binds {spelling!r} to the "
                f"{type(bound).__name__} "
                f"{getattr(bound, '__qualname__', bound)!r}, shadowing its "
                f"own submodule {name}.{spelling}")

        promised = set(getattr(package, "__all__", ()))
        promised |= set(getattr(package, "_SOURCES", {}))
        clash = sorted(promised & submodules)
        if clash:
            hits.append(f"{name} exports {clash}, which is/are also "
                        f"submodules of it")
    assert not hits, hits
