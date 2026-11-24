"""Measure the error from using total-matter power for the cold field.

Run this file first with --mode exact and then --mode total, using the
same output directory. The optional rescaled mode uses P_mm/(1-f_nu)^2.
Each command starts a fresh Cobaya/native state.
Only requests for linear delta_nonu power are replaced; background,
neutrino density, total power, nuisance parameters and covariance stay
fixed. The reported distance is delta_model.T @ precision @ delta_model,
not a difference between two fits to the observed data.
"""

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["COBAYA_NOMPI"] = "1"
project = Path(__file__).resolve().parents[1]
root = project.parents[1]
sys.path.insert(0, str(root / "cobaya"))
sys.path.insert(0, str(root / "external_modules/code/cosmolike_core"))
sys.path.insert(0, str(project / "interface"))

import numpy as np
from cobaya.model import get_model
from cobaya.yaml import yaml_load_file, yaml_dump
import cosmolike_des_cluster_interface as ci

def eh_cold_power_ratio(k, z, omega_m, h, fraction, species, temperature=2.7255):
    """Return (T_cb/T_cbnu)^2 from Eisenstein and Hu's mixed-DM fit.

    k is in physical 1/Mpc and z is an array. Both fields have the SAME
    massive-neutrino cosmology. The authors' power.c factors each transfer
    into tf_master * growth_field / growth_k0, so tf_master and growth_k0
    cancel. The common growth_k0**(1-p_cb) factor also cancels below.
    Only the different scale-dependent growth factors remain.

    Source: https://background.uchicago.edu/~whu/transfer/power.c
    See TFmdm_set_cosm and TFmdm_onek_mpc; paper astro-ph/9710252,
    equations 12--14. We use CAMB's temperature rather than the old
    hardcoded 2.728 K. This test uses flat LCDM, as required by this
    background-growth expression; it is not a general w0/wa prescription.
    """
    if fraction == 0:
        return np.ones((len(z), len(k)))
    theta = temperature / 2.7
    equality = 25000 * omega_m * h**2 / theta**4
    denominator = omega_m * (1 + z)**3 + (1 - omega_m)
    matter_z = omega_m * (1 + z)**3 / denominator
    lambda_z = (1 - omega_m) / denominator
    growth = equality / (1 + z) * 2.5 * matter_z / (
        matter_z**(4 / 7) - lambda_z
        + (1 + matter_z / 2) * (1 + lambda_z / 70)
    )
    exponent = (5 - np.sqrt(1 + 24 * (1 - fraction))) / 4
    q = k * theta**2 / (omega_m * h**2)
    free_streaming = (17.2 * fraction * (1 + 0.488 * fraction**(-7 / 6))
                      * (species * q / fraction)**2)
    transition = (growth[:, None] / (1 + free_streaming[None, :]))**0.7
    # The square converts the transfer-amplitude ratio to a power ratio.
    return ((1 + transition) /
            ((1 - fraction)**(0.7 / exponent) + transition))**(2 * exponent / 0.7)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["exact", "total", "rescaled", "eh"], required=True)
    parser.add_argument("--configuration", default="EXAMPLE_EVALUATE1.yaml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(root)
    info = yaml_load_file(str(project / args.configuration))
    fiducial = dict(info.pop("sampler")["evaluate"]["override"])
    info.pop("output", None)
    info["params"]["mnu"] = {"value": 0.06}
    fiducial.pop("mnu", None)
    # The exact same physical calculation feeds both routes. Refining
    # CAMB avoids confusing a spectrum-computation error with this test.
    camb = info["theory"]["camb"]["extra_args"]
    camb["AccuracyBoost"] = 3.0
    camb["k_per_logint"] = 50
    name = next(iter(info["likelihood"]))
    info["likelihood"][name]["print_datavector"] = False
    configuration = yaml_dump(info)
    if args.mode != "exact":
        assert configuration == (output / "exact_configuration.yaml").read_text()
    (output / (args.mode + "_configuration.yaml")).write_text(configuration)
    points = [fiducial, dict(fiducial, omegam=0.25), dict(fiducial, omegam=0.40)]
    np.savetxt(output / "points.txt", [[p[k] for k in fiducial] for p in points],
               header=" ".join(fiducial))
    model = get_model(info)
    likelihood = model.likelihood[name]
    original = model.provider.get_Pk_interpolator
    cold_requests = []

    def read_power(var_pair=("delta_tot", "delta_tot"), **kwargs):
        # This test-only substitution intercepts the provider request,
        # not the likelihood or native implementation. A new process for
        # each route prevents a cached native table hiding the change.
        if tuple(var_pair) == ("delta_nonu", "delta_nonu"):
            assert kwargs.get("nonlinear", False) is False
            cold_requests.append(args.mode)
            if args.mode == "total":
                var_pair = ("delta_tot", "delta_tot")
        return original(var_pair=var_pair, **kwargs)

    model.provider.get_Pk_interpolator = read_power
    if args.mode in ("rescaled", "eh"):
        def rescaled_neutrino_inputs(lnPL, h):
            # Same approximation as this likelihood's EMUL2 route, but
            # retain CAMB everywhere else to isolate its error alone.
            cold_requests.append(args.mode)
            omega_nu_h2 = model.provider.get_param("omnuh2")
            fraction = omega_nu_h2 / (model.provider.get_param("omegam") * h**2)
            if args.mode == "rescaled":
                return omega_nu_h2, lnPL - 2 * np.log1p(-fraction)
            # log10k_interp_2D is the physical k grid on this likelihood
            # route. Flatten exactly as lnPL does: z index runs fastest.
            ratio = eh_cold_power_ratio(
                k=10.0**likelihood.log10k_interp_2D,
                z=likelihood.z_interp_2D,
                omega_m=model.provider.get_param("omegam"),
                h=h,
                fraction=fraction,
                species=camb["num_massive_neutrinos"],
                temperature=camb.get("TCMB", 2.7255),
            )
            return omega_nu_h2, lnPL + np.log(ratio).flatten(order="F")

        likelihood.get_neutrino_inputs = rescaled_neutrino_inputs
    np.savetxt(output / "blocks.txt",
               np.column_stack((ci.compute_data_vector_cluster_starts(),
                                ci.compute_data_vector_cluster_sizes())),
               fmt="%d", header="start size; rows: ss gs gg cg N cc cs")
    vectors = []
    full_vectors = []
    for index, point in enumerate(points):
        posterior = model.logposterior(params_values=point, cached=False)
        if not np.isfinite(posterior.loglike):
            raise ValueError("Likelihood failed; no previous vector may be reused.")
        inputs = model.parameterization.to_input(
            sampled_params_values=model.parameterization.check_sampled(point),
        )
        full = likelihood.get_datavector(**inputs)
        vector = full
        precision = np.asarray(ci.get_inv_cov_masked_cluster())
        assert precision.shape == (len(vector), len(vector))
        assert np.all(np.isfinite(vector))
        full_vectors.append(full.copy())
        vectors.append(vector.copy())
        # Inspect the two true CAMB spectra even in the substitution run.
        # These k values are physical 1/Mpc, with no h rescaling.
        k = np.geomspace(1e-4, 10.0, 301)
        z = np.array([0.0, 0.5, 1.0, 2.0])
        cold = original(var_pair=("delta_nonu", "delta_nonu"), nonlinear=False)
        total = original(var_pair=("delta_tot", "delta_tot"), nonlinear=False)
        ratio = total.P(z, k) / cold.P(z, k) - 1.0
        np.savetxt(output / (args.mode + f"_power_{index}.txt"),
                   np.column_stack((k, ratio.T)), header="k_per_Mpc Pmm_over_Pcb_minus1_z0 z0p5 z1 z2")
        fraction = model.provider.get_param("omnuh2") / (point["omegam"] * (point["H0"] / 100)**2)
        eh_ratio = eh_cold_power_ratio(
            k=k, z=z, omega_m=point["omegam"], h=point["H0"] / 100,
            fraction=fraction, species=camb["num_massive_neutrinos"],
            temperature=camb.get("TCMB", 2.7255),
        )
        np.savetxt(output / (args.mode + f"_eh_power_residual_{index}.txt"),
                   np.column_stack((k, ((1 + ratio) * eh_ratio - 1).T)),
                   header="k_per_Mpc Pmm_times_EH_over_Pcb_minus1_z0 z0p5 z1 z2")
        print(args.mode, index, "f_nu", fraction,
              "power_fraction_range", ratio.min(), ratio.max(), flush=True)
    assert cold_requests, "The test must actually exercise a cold-power request."
    np.save(output / (args.mode + "_vectors.npy"), vectors)
    np.save(output / (args.mode + "_full_vectors.npy"), full_vectors)
    if args.mode == "exact":
        np.save(output / "precision.npy", precision)
    else:
        np.testing.assert_array_equal(precision, np.load(output / "precision.npy"))
        reference = np.load(output / "exact_vectors.npy")
        difference = np.asarray(vectors) - reference
        distances = np.einsum("ni,ij,nj->n", difference, precision, difference)
        np.savetxt(output / (args.mode + "_distances.txt"), distances, header="direct_vector_delta_chi2")
        print("Direct-vector distances:", distances, flush=True)
    native = Path(ci.__file__)
    record = ["mode " + args.mode, "cold_requests " + str(len(cold_requests)),
              "native_sha256 " + hashlib.sha256(native.read_bytes()).hexdigest()]
    for repo in [project, root / "external_modules/code/cosmolike_core", root / "external_modules/code/CAMB"]:
        revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        record.append(str(repo) + " " + revision)
    (output / (args.mode + "_provenance.txt")).write_text("\n".join(record) + "\n")
    model.close()
