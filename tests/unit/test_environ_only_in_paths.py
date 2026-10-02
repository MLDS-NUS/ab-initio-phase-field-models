"""Only ``aipf.paths`` and ``aipf.site`` read configuration from the environment.

Four modules touch ``os.environ`` for another reason: they build or clean a child process's
environment, inspect the variables a run will inherit, or save and restore one variable around a
call. Each of their lines must be one of those forms, never a configuration read.
"""
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "aipf"
ALLOWED = {SRC / "paths.py", SRC / "site.py"}
PATTERN = re.compile(r"os\.environ|getenv\(")

#: The non-configuration forms, per allowed writer.
_WRITER_FORMS = {
    SRC / "train" / "fit.py": (
        r"os\.environ\[\w+\]\s*=",                 # restore the saved value
        r"os\.environ\.pop\(",                     # restore its absence
        r"\w+ = os\.environ\.get\(_CUBLAS_VAR\)",  # save it first
    ),
    SRC / "md" / "engine.py": (
        r"= os\.environ if env is None else env",  # the environment a child inherits, or the caller's
    ),
    SRC / "md" / "doctor.py": (
        r"dict\(os\.environ if env is None else env\)",
        r"= os\.environ if environ is None else environ",  # the variables a run would inherit
        r'os\.environ\.get\("LD_LIBRARY_PATH", ""\)',      # inside the probe run by the interpreter under test
    ),
    SRC / "md" / "scheduler.py": (
        r"= os\.environ\.copy\(\)",                # the job's environment, then its declared changes
    ),
}


def test_only_paths_and_site_read_the_environment():
    hits = [p for p in SRC.rglob("*.py") if p not in ALLOWED and PATTERN.search(p.read_text())]
    # the two legitimate writers: fit's CUBLAS restore and the subprocess environment builders in md/
    allowed_writers = {SRC / "train" / "fit.py", SRC / "md" / "engine.py", SRC / "md" / "doctor.py", SRC / "md" / "scheduler.py"}
    assert set(hits) <= allowed_writers, [str(h) for h in hits]


def test_the_allowed_writers_only_build_or_inspect_a_process_environment():
    stray = []
    for path, forms in _WRITER_FORMS.items():
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if PATTERN.search(line) and not any(re.search(form, line) for form in forms):
                stray.append(f"{path.relative_to(SRC)}:{number}: {line.strip()}")
    assert not stray, stray
