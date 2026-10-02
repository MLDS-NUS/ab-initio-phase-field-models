import declared_roots
from aipf.cli.main import main


def test_data_subcommand_is_registered(capsys):
    # argparse's own ``-h``/``--help`` action always calls ``parser.exit()``,
    # which raises ``SystemExit`` even when invoked in-process through
    # ``main()`` -- the brief's own second case below already accounts for
    # this same behaviour on a parse error, so the first case is brought in
    # line with it rather than asserting a return value ``--help`` can never
    # produce.
    try:
        main(["--help"])
    except SystemExit as e:
        assert e.code == 0
    assert "data" in capsys.readouterr().out


def test_data_build_requires_a_system(capsys):
    try:
        main(["data", "build"])
    except SystemExit as e:
        assert e.code != 0
    assert "--system" in capsys.readouterr().err


def test_data_rebuild_requires_a_system(capsys):
    try:
        main(["data", "rebuild"])
    except SystemExit as e:
        assert e.code != 0
    assert "--system" in capsys.readouterr().err


def test_data_verify_is_gone(capsys):
    """The fingerprint check of the read-only trees is not part of the package."""
    try:
        main(["data", "verify", "--manifest", "m.json"])
    except SystemExit as e:
        assert e.code != 0
    assert "invalid choice" in capsys.readouterr().err


def test_data_build_refuses_an_undeclared_raw_root(capsys, monkeypatch):
    """No traceback: the refusal names the variable and the aipf.toml key."""
    from aipf import paths
    monkeypatch.delenv("AIPF_RAW_FEB", raising=False)
    monkeypatch.delenv("AIPF_RAW", raising=False)
    monkeypatch.setattr(paths, "config", lambda: {})
    assert main(["data", "build", "--system", "feb", "--dry-run"]) == 2
    err = capsys.readouterr().err
    assert "AIPF_RAW_FEB" in err and "[paths.raw]" in err and "aipf.toml" in err


def test_data_build_refuses_a_raw_root_that_is_not_there(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("AIPF_RAW_FEB", str(tmp_path / "absent"))
    assert main(["data", "build", "--system", "feb", "--dry-run"]) == 2
    err = capsys.readouterr().err
    assert str(tmp_path / "absent") in err and "AIPF_RAW_FEB" in err and "aipf.toml" in err


def test_an_undeclared_raw_root_is_a_refusal_in_every_command(capsys, monkeypatch, tmp_path):
    """A command that reaches an undeclared raw root (here the dome stage) exits 2 naming the variable
    and the key, not a traceback: the refusal is caught once, in the entry point."""
    declared_roots.published_or_skip("hhe")
    from aipf import paths
    monkeypatch.setattr(paths, "config", lambda: {})
    monkeypatch.delenv("AIPF_RAW_HHE", raising=False)
    monkeypatch.delenv("AIPF_RAW", raising=False)
    assert main(["diagnose", "--system", "hhe", "--ckpt", "published", "--stage", "dome",
                 "--out", str(tmp_path / "out")]) == 2
    err = capsys.readouterr().err
    assert err.startswith("aipf diagnose: ") and "AIPF_RAW_HHE" in err and "[paths.raw]" in err


def test_a_malformed_aipf_toml_is_a_refusal_naming_the_file(capsys, monkeypatch, tmp_path):
    from aipf import paths
    (tmp_path / "aipf.toml").write_text("[paths\n")
    monkeypatch.setattr(paths, "repo_root", lambda: tmp_path)
    monkeypatch.delenv("AIPF_RAW_FEB", raising=False)
    monkeypatch.delenv("AIPF_RAW", raising=False)
    paths.config.cache_clear()
    try:
        assert main(["data", "build", "--system", "feb", "--dry-run"]) == 2
    finally:
        paths.config.cache_clear()
    assert str(tmp_path / "aipf.toml") in capsys.readouterr().err
