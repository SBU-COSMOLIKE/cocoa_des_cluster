"""Check that the independent reference command fails on failed comparisons.

The expensive reference calculations are replaced with short, known tables.
The test exercises the command's actual final acceptance decision, including
writing its diagnostic record before raising an error. Historical comparisons
with deliberately different physics remain advisory.
"""
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation"))
import compare_reference as validation


@pytest.mark.parametrize("failure", [None, "matched", "nuisance", "production", "determinism", "diagnostic"])
def test_required_failures_raise_after_recording(tmp_path, monkeypatch, failure):
    """A bad required row or thread invariant cannot return a successful run."""
    args = SimpleNamespace(
        cache_dir=str(tmp_path), threads=1, skip_diagnostics=False,
        skip_production=False, skip_determinism=False, skip_lighthouse=True,
        json=str(tmp_path / "result.json"),
    )
    reference = dict(cin_mnu=0.06, cin_omnuh2=0.001, cin_omegam=0.3,
                     cin_omegab=0.05, cin_H0=70., label="matched")
    monkeypatch.setattr(validation, "init_c", lambda *a: (None, {}))
    monkeypatch.setattr(validation, "set_nuisance_c", lambda *a: None)
    monkeypatch.setattr(validation, "c_evaluate", lambda *a, **k: {})
    monkeypatch.setattr(validation, "load_reference",
                        lambda *a: (dict(reference, label=a[-1]), "unused"))
    monkeypatch.setattr(validation, "compare_all", lambda c, r, *a, **k: (
        [dict(name="known target", status="MISS" if failure == r["label"] else "ok")], {}))
    monkeypatch.setattr(validation, "print_table", lambda *a: None)
    monkeypatch.setattr(validation, "reference_in_c_layout", lambda *a: (None, None))
    monkeypatch.setattr(validation, "run_determinism", lambda *a: {
        "omp1_vs_omp8": failure != "determinism",
        "omp1_back_to_point": True, "omp8_back_to_point": True,
        "detour_changes_dv": True, "omp1_vs_omp8_detour": True,
        "max_abs_diff_omp1_omp8": 0.0,
    })
    config = {"likelihood": {"cluster_hmf_alpha_mode": 0}}
    if failure in (None, "diagnostic"):
        validation.run_validation(args, config, reference, "unused", str(tmp_path))
    else:
        with pytest.raises(RuntimeError, match="Cluster validation failed"):
            validation.run_validation(args, config, reference, "unused", str(tmp_path))
    assert Path(args.json).is_file()
