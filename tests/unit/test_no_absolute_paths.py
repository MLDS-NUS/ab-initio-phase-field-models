"""No tracked file a newcomer runs or reads names the machine it was written on.

Every location the package, the tests, the experiments, the figures and the setup read
comes from a declared place: ``aipf.paths`` and ``System.paths`` (repository, ``data/``,
the raw roots) and ``aipf.site``, each an ``AIPF_*`` variable or a key of the local and
untracked ``aipf.toml`` (its tracked template is ``aipf.toml.example``;
``docs/guides/environment.md``). ``tests/declared_roots.py`` is the tests' route to
all of them.

The scan covers every text file under ``ROOTS``, comments and docstrings included, for
any rooted ``home`` or ``scratch`` directory with a named directory below it, which covers
the usual cluster prefixes without spelling them. No file is exempt.
"""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

ROOTS = ("src", "tests", "experiments", "figures", "env",
         "docs/reference", "docs/guides",
         "README.md", "pyproject.toml", ".gitignore", "aipf.toml.example")

#: A rooted home or scratch directory with a name under it.
HOST = re.compile(r"/(home|scratch)/[^/\s'\"`]+/")
#: A quoted container root, the one form the earlier rule also refused.
QUOTED_APP = re.compile(r"""['"]/app/""")

_SKIP_DIRS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
_SKIP_UNDER = ("figures/out/",)


def _files(repo: Path = REPO):
    for root in ROOTS:
        base = repo / root
        if base.is_file():
            yield base
            continue
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            rel = p.relative_to(repo).as_posix()
            if (not p.is_file() or p.is_symlink() or p.suffix == ".pyc"
                    or any(part in _SKIP_DIRS or part.endswith(".egg-info") for part in p.parts)
                    or rel.startswith(_SKIP_UNDER)):
                continue
            yield p


def host_path_hits(repo: Path = REPO) -> list[str]:
    hits = []
    for p in _files(repo):
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue        # a binary input (figures/<fig>/figdata, figures/reference)
        rel = p.relative_to(repo).as_posix()
        for number, line in enumerate(text.splitlines(), 1):
            if HOST.search(line) or (p.suffix == ".py" and QUOTED_APP.search(line)):
                hits.append(f"{rel}:{number}: {line.strip()[:100]}")
    return hits


def test_no_tracked_file_names_a_home_or_scratch_directory():
    hits = host_path_hits()
    assert not hits, "\n".join(hits)


def test_the_scan_sees_the_trees_it_names():
    """A scan that found no file would pass forever."""
    seen = {p.relative_to(REPO).as_posix().split("/")[0] for p in _files()}
    assert {"src", "tests", "experiments", "figures", "env", "README.md"} <= seen, seen


def test_the_pattern_fires_on_a_host_path_and_not_on_prose():
    assert HOST.search("prefix: " + "/" + "scratch/someone/envs/aipf")
    assert HOST.search("cd " + "/" + "home/someone/project")
    assert not HOST.search("src/aipf/paths.py and data/lj/ckpt/published/final.ckpt")
    assert not HOST.search("$AIPF_RAW/hhe/runs")


def test_a_host_path_in_a_figure_directory_is_found(tmp_path):
    """figures/ is scanned like every other root, data files included."""
    fig = tmp_path / "figures" / "some_fig"
    (fig / "figdata").mkdir(parents=True)
    (fig / "draw.py").write_text('"""A figure."""\n')
    (fig / "figdata" / "notes.json").write_text('{"src": "' + "/" + 'scratch/someone/run/"}\n')
    assert host_path_hits(tmp_path) == [
        "figures/some_fig/figdata/notes.json:1: " + (fig / "figdata" / "notes.json").read_text().strip()]
