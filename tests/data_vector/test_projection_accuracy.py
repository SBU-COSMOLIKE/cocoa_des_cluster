"""Measure radial quadrature and CAMB convergence in the live DES Cluster examples.

The 4×2pt+N and 6×2pt+N masks both retain galaxy clustering. Each
comparison keeps the cosmology, nuisance parameters, data and precision
matched. E is the squared prediction difference weighted by the full
masked precision. The signed change in chi-squared against the supplied
data is reported separately. The combined comparison uses the higher
CAMB controls and native orders together and checks the total accuracy
target E < 0.2. The one-control comparisons report numerical distances
and check matched inputs, finite scores and independent probe controls.

Run from activated Cocoa with pytest -s. Each test owns a child process.
"""
from contextlib import chdir
from copy import deepcopy
import os
from pathlib import Path
import sys
import unittest

os.environ["OMP_NUM_THREADS"] = os.environ.get("COCOA_TEST_OMP_THREADS", "4")
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["COBAYA_NOMPI"] = "1"

import numpy as np

project = Path(__file__).resolve().parents[2]
root = project.parents[1]
sys.path.insert(0, str(project / "tests"))
import cocoa_test_utils as u

probes = ("ss", "gs", "gg", "cg", "n", "cc", "cs")


def _configuration(example):
    """Read the live likelihood and its evaluation fiducial."""
    from cobaya.yaml import yaml_load_file

    info = yaml_load_file(str(project / f"EXAMPLE_EVALUATE{example}.yaml"))
    center = info.pop("sampler")["evaluate"]["override"]
    info.pop("output", None)
    info["timing"] = False
    return info, center


def _points(model, center):
    """Use the fiducial and move sampled parameters on either side of it."""
    names = model.parameterization.sampled_params()
    fiducial = {name: center[name] for name in names}
    points = {"fiducial": fiducial}
    for label, sign in (("lower H0", -1), ("higher H0", 1)):
        point = {}
        for name, value in fiducial.items():
            offset = 0.015 * abs(value) if value != 0 else 0.001
            point[name] = value + sign * offset
        point["H0"] = fiducial["H0"] + sign * 8.
        points[label] = point
    return points


def _survey(ci):
    """Read the joint mask, data and metric in the native block order."""
    mask = np.array(ci.get_mask_cluster(), dtype=bool)
    sizes = ci.compute_data_vector_cluster_sizes()
    edges = np.r_[0, np.cumsum(sizes)]
    assert len(sizes) == len(probes) and edges[-1] == len(mask)
    blocks = {}
    for index, probe in enumerate(probes):
        blocks[probe] = edges[index], edges[index + 1]
    start, stop = blocks["gg"]
    assert mask[start:stop].any(), "This example must retain galaxy clustering."
    data = np.array(ci.get_dv_masked_cluster())[mask]
    precision = np.array(ci.get_inv_cov_masked_cluster())[np.ix_(mask, mask)]
    return mask, data, precision, blocks


def _prediction(model, ci, point, mask, data, precision):
    """Verify the native vector reproduces the likelihood at this point."""
    result = model.logposterior(point, cached=False)
    assert np.isfinite(result.logpost), point
    vector = np.array(ci.compute_data_vector_cluster_masked(), copy=True)
    assert vector.shape == mask.shape and np.isfinite(vector).all()
    score = -2. * result.loglike
    residual = vector[mask] - data
    np.testing.assert_allclose(score, residual @ precision @ residual, rtol=1e-12, atol=1e-9)
    return vector, score


def _report(label, baseline, refined, mask, precision):
    """Report the prediction distance and baseline-minus-refined fit change."""
    delta = (baseline[0] - refined[0])[mask]
    distance = delta @ precision @ delta
    change = baseline[1] - refined[1]
    assert np.isfinite(distance) and distance >= 0. and np.isfinite(change)
    print(f"{label:50s}  E={distance:.6g}  observed Δχ²={change:+.6g}", flush=True)
    return distance


def _native_settings(like):
    """Read the likelihood's production boost, level and projection factors."""
    settings = {
        "accuracy_boost": like.accuracyboost,
        "integration_accuracy": like.integration_accuracy,
    }
    for probe in probes:
        name = f"integration_accuracy_{probe}"
        settings[name] = getattr(like, name, 1.)
    return settings


def _check_native(example):
    """Compare orders at fixed CAMB tables and check the changed block."""
    from cobaya.model import get_model
    import cosmolike_des_cluster_interface as ci

    info, center = _configuration(example)
    like_name, = info["likelihood"]
    with chdir(root), get_model(info) as model:
        like = model.likelihood[like_name]
        assert like.use_emulator == 0 and like.integration_accuracy == 1
        native = _native_settings(like)
        mask, data, precision, blocks = _survey(ci)
        for label, point in _points(model, center).items():
            baseline = _prediction(model, ci, point, mask, data, precision)
            for probe, (start, stop) in blocks.items():
                if not mask[start:stop].any():
                    continue
                key = f"integration_accuracy_{probe}"
                settings = dict(native)
                settings[key] *= 2.
                ci.init_accuracy_boost(**settings)
                changed = np.array(ci.compute_data_vector_cluster_masked(), copy=True)
                np.testing.assert_array_equal(changed[:start], baseline[0][:start])
                np.testing.assert_array_equal(changed[stop:], baseline[0][stop:])
                assert not np.array_equal(changed[start:stop], baseline[0][start:stop])
                refined = changed, ci.compute_chi2_cluster(changed)
                _report(f"EX{example} {label} {probe} doubled order",
                        baseline, refined, mask, precision)

            ci.init_accuracy_boost(**dict(
                native, integration_accuracy=3, integration_accuracy_n=2.,
            ))
            vector = np.array(ci.compute_data_vector_cluster_masked(), copy=True)
            reference = vector, ci.compute_chi2_cluster(vector)
            _report(f"EX{example} {label} ordinary vs higher native orders",
                    baseline, reference, mask, precision)
            ci.init_accuracy_boost(**native)
            np.testing.assert_array_equal(ci.compute_data_vector_cluster_masked(), baseline[0])


def _check_camb(example):
    """Check CAMB spacing, transfer accuracy and coverage at ordinary native orders."""
    from cobaya.model import get_model
    import cosmolike_des_cluster_interface as ci

    config, center = _configuration(example)
    like_name, = config["likelihood"]
    trials = {
        "ordinary": ({}, None),
        "spacing": ({"k_per_logint": 25}, None),
        "transfer boost": ({"AccuracyBoost": 1.5}, None),
        "coverage": ({"kmax": 20.}, 20.),
        "neutrino momenta": ({"Accuracy.neutrino_q_boost": 1.25}, None),
        "joint": ({
            "k_per_logint": 25,
            "AccuracyBoost": 1.5,
            "kmax": 20.,
            "Accuracy.neutrino_q_boost": 1.25,
        }, 20.),
    }
    survey_metric = None
    results = {}
    combined = {}
    with chdir(root):
        for trial, (overrides, native_kmax) in trials.items():
            info = deepcopy(config)
            results[trial] = {}
            if native_kmax is not None:
                info["likelihood"][like_name]["kmax_boltzmann"] = native_kmax
            camb = info["theory"]["camb"]
            camb["stop_at_error"] = True
            requested = camb["extra_args"]
            requested.update(overrides)
            with get_model(info) as model:
                like = model.likelihood[like_name]
                assert like.use_emulator == 0 and like.integration_accuracy == 1
                mask, data, precision, _ = _survey(ci)
                native = _native_settings(like)
                if survey_metric is None:
                    survey_metric = mask, data, precision
                else:
                    for values, reference in zip((mask, data, precision), survey_metric, strict=True):
                        np.testing.assert_array_equal(values, reference)
                for label, point in _points(model, center).items():
                    prediction = _prediction(model, ci, point, mask, data, precision)
                    _, transfers = model.theory["camb.transfers"].get_CAMB_transfers()
                    params = transfers.Params
                    boost = params.Accuracy.AccuracyBoost
                    q_boost = params.Accuracy.neutrino_q_boost
                    assert params.num_nu_massive == requested["num_massive_neutrinos"]
                    assert boost == requested["AccuracyBoost"]
                    assert q_boost == requested.get("Accuracy.neutrino_q_boost", 1.)
                    assert params.Transfer.k_per_logint == requested["k_per_logint"]
                    effective_kmax = max(requested["kmax"], like.kmax_boltzmann * like.accuracyboost)
                    if params.NonLinear in ("NonLinear_lens", "NonLinear_both") and params.DoLensing:
                        effective_kmax = max(effective_kmax, 5. * boost * params.Accuracy.NonlinSourceBoost)
                    assert params.Transfer.kmax == effective_kmax
                    print(f"EX{example} {label} {trial}: CAMB boost={boost:g} "
                          f"k_per_logint={params.Transfer.k_per_logint} "
                          f"effective kmax={effective_kmax:g} neutrino_q_boost={q_boost:g}")
                    results[trial][label] = prediction
                    if trial == "joint":
                        # Reuse the higher CAMB fields for the native refinement.
                        ci.init_accuracy_boost(**dict(
                            native, integration_accuracy=3, integration_accuracy_n=2.,
                        ))
                        vector = np.array(ci.compute_data_vector_cluster_masked(), copy=True)
                        combined[label] = vector, ci.compute_chi2_cluster(vector)
                        ci.init_accuracy_boost(**native)

    for label, prediction in combined.items():
        distance = _report(f"EX{example} {label} combined native and CAMB",
                           results["ordinary"][label], prediction, mask, precision)
        assert distance < 0.2, f"EX{example} {label}: combined E = {distance:.6g} must be below 0.2"
    for trial, predictions in results.items():
        for label, prediction in predictions.items():
            if trial != "ordinary":
                _report(f"EX{example} {label} CAMB ordinary → {trial}",
                        results["ordinary"][label], prediction, mask, precision)
            if trial not in ("ordinary", "joint"):
                _report(f"EX{example} {label} CAMB {trial} → joint",
                        prediction, results["joint"][label], mask, precision)


class TestProjectionAccuracy(unittest.TestCase):
    """Exercise the ordinary radial resolutions on both joint masks."""

    @u.own_process(__file__)
    def test_native_4x2ptn(self):
        """Check gg and the active cluster blocks of the 4×2pt+N likelihood."""
        _check_native(1)

    @u.own_process(__file__)
    def test_native_6x2ptn(self):
        """Check the full 6×2pt+N projection, including all six lens bins."""
        _check_native(2)

    @u.own_process(__file__)
    def test_camb_4x2ptn(self):
        """Refine CAMB against the ordinary 4×2pt+N likelihood."""
        _check_camb(1)

    @u.own_process(__file__)
    def test_camb_6x2ptn(self):
        """Refine CAMB against the ordinary 6×2pt+N likelihood."""
        _check_camb(2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
