"""``MANIFEST.json``'s ``kernel_hinge`` says at what ``lambda_W`` the run trained the hinge, not only
its shape (the weight used to be in ``hparams.yaml`` alone)."""
from aipf.system import load
from aipf.train.config import TrainConfig
from aipf.train.fit import _config_from_system
from aipf.train.kernel_hinge import KernelHinge


def test_provenance_records_the_weight():
    hinge = KernelHinge.from_config(TrainConfig(**_config_from_system(load("feb"))))
    assert hinge.provenance() == {"term": "L_W", "kappa": 2.0, "k_max": 3.0, "n_k": 32,
                                  "margin": 0.0, "lambda_W": 0.3}


def test_the_weight_is_recorded_not_compared():
    shape = dict(kappa=2.0, k_max=3.0, n_k=32, margin=0.0)
    assert KernelHinge(**shape) == KernelHinge(**shape, weight=0.3)
    assert KernelHinge(**shape).provenance()["lambda_W"] is None
