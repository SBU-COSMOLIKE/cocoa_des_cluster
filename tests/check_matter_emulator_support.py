"""Check whether the candidate matter emulator supplies the cluster fields.

The DES cluster likelihood uses total-matter power (delta_tot) for lensing
and cold-matter power (delta_nonu: baryons plus cold dark matter) for the
halo mass function. Its current EMUL2 route approximates the cold spectrum
as P_lin/(1-f_nu)^2. That changes the physical inputs, so the matched
50-point comparison is blocked until the provider supplies cold power.
This small Cobaya test asks the candidate provider for both fields at
fixed mnu=0.06 eV, before an expensive random-cosmology comparison.

Run from activated Cocoa, using EMULATOR_CODE_DIR and EMULATOR_DATA_DIR to
select the candidate code and model checkouts:
    python projects/des_cluster/tests/check_matter_emulator_support.py

The output goes under chains/matter_emulator_prior/. Exit status 1 means
that the provider does not supply a required field; it is not an accuracy
result and no substitute spectrum is introduced.
"""

import os
from pathlib import Path
import sys

# Keep this support check small, including when other project tests run.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["COBAYA_NOMPI"] = "1"

project = Path(__file__).resolve().parents[1]
root = project.parents[1]
code = Path(os.environ.get(
    "EMULATOR_CODE_DIR",
    root / "external_modules/code/emulators/emultrf",
))
data = Path(os.environ.get(
    "EMULATOR_DATA_DIR",
    root / "external_modules/data/emultrf",
))
sys.path.insert(0, str(root / "cobaya"))
sys.path.insert(0, str(code))

import numpy as np
from cobaya.log import LoggedError
from cobaya.model import get_model
from emulmps.emulmps import emulmps


if __name__ == "__main__":
    model = get_model(info_or_yaml_or_file={
        "params": {
            "As_1e9": 2.1,
            "ns": 0.96,
            "H0": 67.0,
            "omegab": 0.049,
            "omegam": 0.319,
            "w": -1.0,
            "wa": 0.0,
        },
        "likelihood": {"one": None},
        "theory": {
            "emulmps": {
                "external": emulmps,
                "stop_at_error": True,
                "extra_args": {
                    "model_file": str(root / "external_modules/data/emultrf/emulmps/npce_emul/emulator_npce.keras"),
                    "metadata_file": str(root / "external_modules/data/emultrf/emulmps/npce_emul/metadata.joblib"),
                    "nl_model_file": str(data / "emulmps/w0wa_mead2020/emulator_mead2020.keras"),
                    "nl_metadata_file": str(data / "emulmps/w0wa_mead2020/metadata.joblib"),
                    "param_order": ["As_1e9", "ns", "H0", "omegab", "omegam", "w", "wa"],
                    "fixed_params": {"mnu": 0.06},
                },
            },
        },
    })
    # These are the two field pairs requested by the production cluster
    # likelihood. No halo or bispectrum calculation is needed to test
    # whether the power provider implements this prerequisite.
    model.add_requirements(requirements={
        "Pk_interpolator": {
            "z": [0.0, 0.5, 1.0, 2.0],
            "k_max": 5.0,
            "nonlinear": (False, True),
            "vars_pairs": [
                ("delta_tot", "delta_tot"),
                ("delta_nonu", "delta_nonu"),
            ],
        },
    })
    posterior = model.logposterior(params_values={}, cached=False)
    if not np.isfinite(posterior.loglike):
        raise RuntimeError("Candidate failed at the support-check cosmology.")
    lines = []
    failed = 0
    for field in ("delta_tot", "delta_nonu"):
        try:
            power = model.provider.get_Pk_interpolator(
                var_pair=(field, field),
                nonlinear=False,
            )
            values = power.P([0.0, 0.5], [0.01, 0.1])
            if not np.all(np.isfinite(values)):
                raise ValueError("Provider returned nonfinite power.")
            lines.append(f"{field}: supplied finite linear power")
        except (LoggedError, ValueError) as error:
            lines.append(f"{field}: UNSUPPORTED: {error}")
            failed += 1
    output = project / "chains/matter_emulator_prior"
    output.mkdir(parents=True, exist_ok=True)
    (output / "support.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    model.close()
    sys.exit(1 if failed else 0)
