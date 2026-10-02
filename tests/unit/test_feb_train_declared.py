"""Iron-boron trains from its declaration alone: ``aipf train`` four steps, anchors declared.

Two cube runs of the published model's own 0 GPa mode tree, one per phase class,
each under the declared source that holds it in the published partition
(``defaults["training"]["sources"]``), which ``source_loss_weights`` weighs. The run
starts from the published model's weights with a fresh optimizer, so the drift
term is the trained model's and not a random one's. ~50 s, one thread.
"""
import json
import math

import pytest

from aipf.system import load

pytestmark = [pytest.mark.slow, pytest.mark.env]

def _source(name, tag):
    """``NAME=TREE:TAG:32,32,32``; TREE is the published model's ``modes_root``, one run per source."""
    tree = load("feb").constants["modes_trees"][0]
    return f"{name}={tree}:{tag}:32,32,32"


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    import os

    from aipf.cli.main import main

    import declared_roots
    system = load("feb")
    tree = declared_roots.raw_or_skip("feb", "fields", system.constants["modes_trees"][0])
    root = tmp_path_factory.mktemp("data")
    old = os.environ.get("AIPF_DATA")
    os.environ["AIPF_DATA"] = str(root)
    try:
        code = main(["train", "--system", "feb", "--run", "c1_four_steps",
                     "--seed", "0", "--steps", "4",
                     "--source", _source("noneq_0", "cube_x0.95_T1700_s1"),
                     "--source", _source("stable_0", "cube_x0.50_T2300_s1"),
                     "--init-from-published", "--resume-optimizer", "no",
                     "--anchors", "declared", "--log-every-step"])
    finally:
        if old is None:
            os.environ.pop("AIPF_DATA", None)
        else:
            os.environ["AIPF_DATA"] = old
    assert code == 0
    return root / "feb" / "ckpt" / "c1_four_steps"


def test_the_run_writes_its_checkpoint_and_its_manifest(run_dir):
    assert (run_dir / "final.ckpt").is_file()
    manifest = json.loads((run_dir / "MANIFEST.json").read_text())
    assert manifest["system"] == "feb"
    import torch
    assert manifest["device"] == ("cuda" if torch.cuda.is_available() else "cpu")  # --device auto
    assert manifest["global_step"] == 4
    assert manifest["resume_optimizer"] is False
    assert manifest["init_from"]["md5"] == load("feb").checkpoint.md5
    assert [s["name"] for s in manifest["sources"]] == ["noneq_0", "stable_0"]
    assert {s["loss_weight"] for s in manifest["sources"]} == {15000.0}


def test_every_declared_anchor_term_is_trained_on_the_published_models_rows(run_dir):
    manifest = json.loads((run_dir / "MANIFEST.json").read_text())
    # and the kernel hinge the declaration weighs at lambda_wpsd = 0.3
    assert manifest["terms_trained"] == ["L_dyn", "L_M", "L_S", "L_bulk",
                                         "L_P", "L_W"]
    assert manifest["declared_weights_without_data"] == {}
    assert manifest["kernel_hinge"] == {"term": "L_W", "kappa": 2.0, "k_max": 3.0,
                                        "n_k": 32, "margin": 0.0, "lambda_W": 0.3}
    assert manifest["anchor_tables"]["rows"] == {
        "anchor_M": 288, "anchor_S": 286, "anchor_bulk": 286, "anchor_P": 442}
    assert not (run_dir / "UNTRAINED_TERMS.txt").exists()


def test_every_step_is_finite_including_the_zero_pressure_term(run_dir):
    """The 0 GPa manifold's target is zero; the declared floor keeps L_P finite."""
    steps = json.loads((run_dir / "steps.json").read_text())
    assert len(steps["loss"]) == 4
    for loss, terms in zip(steps["loss"], steps["terms"]):
        assert math.isfinite(loss)
        assert all(math.isfinite(v) for v in terms.values()), terms


def test_the_run_used_the_declared_estimator_and_order(run_dir):
    manifest = json.loads((run_dir / "MANIFEST.json").read_text())
    assert manifest["split_mode"] == "random"
    assert manifest["loader_order"] == "shuffled"
    assert load("feb").defaults["estimator"] == "weak_mid"


def test_each_smoke_run_sits_in_the_declared_source_it_is_named_for():
    """The two runs above are a plumbing set, but each is filed where the published partition files it."""
    sources = load("feb").defaults["training"]["sources"]
    for name, tag in (("noneq_0", "cube_x0.95_T1700_s1"), ("stable_0", "cube_x0.50_T2300_s1")):
        assert tag not in sources[name]["exclude_tags"], (name, tag)
        others = [n for n, s in sources.items()
                  if n != name and s["root"] == sources[name]["root"]]
        assert all(tag in sources[n]["exclude_tags"] for n in others), (name, tag)
