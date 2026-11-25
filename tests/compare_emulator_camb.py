"""Compare the candidate matter emulator with CAMB at random des_cluster cosmologies.

Run from activated Cocoa (one project at a time):
    python projects/des_cluster/tests/compare_emulator_camb.py
    mpirun -n 2 --bind-to none python -m mpi4py \
      projects/des_cluster/tests/compare_emulator_camb.py --mpi

MPI assigns independent parameter columns or cosmologies to each rank;
OpenMP accelerates each rank's model evaluations. Keep BLAS at one thread.

The shared Cocoa test uses the numbered example's cosmological priors,
restricted to 0.2 < Omega_m < 0.5 and fixed mnu=0.06 eV. Nuisance parameters
stay at its declared point. Both calculations use plain mead2020 and CAMB
background distances. This does not change the production nonlinear
prescription or its sampled neutrino-mass prior.

Use EMULATOR_CODE_DIR and EMULATOR_DATA_DIR to select candidate checkouts.
COCOA_TEST_INTERFACE_DIR optionally selects an independently built native
interface. Results, including failed points, go under chains/.

The cluster test fixes w=-1 and wa=0 for the validated cold-power fit.
It verifies that the amplitude cut is redundant over the full prior box
before removing that external prior from this test only.

The optional --points argument reads explicit named-column parameter points.
The default Fisher calculation varies the sampled cosmological and nuisance
parameters for this project's full retained observable. The shared test
then selects 25 points from its bounded Fisher distribution and evaluates
both theories again. --local-points replaces that selection with a named
parameter table. The E5 test uses --advisory and retains all outcomes.
"""

import argparse
import os
from pathlib import Path
import sys

# Set thread ownership before NumPy, TensorFlow or the native library loads.
os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "4")
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = os.environ["OMP_NUM_THREADS"]
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["COBAYA_NOMPI"] = "1"

project = Path(__file__).resolve().parents[1]
root = project.parents[1]
core = root / "external_modules/code/cosmolike_core"
interface = Path(os.environ.get("COCOA_TEST_INTERFACE_DIR", project / "interface"))
sys.path.insert(0, str(core))
sys.path.insert(0, str(interface))
sys.path.insert(0, str(root / "cobaya"))

import numpy as np
import cosmolike_des_cluster_interface as ci
from cocoa_testing import (
    run_matter_emulator_prior_check,
    plot_matter_emulator_fisher,
    sample_matter_emulator_fisher,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", default="EXAMPLE_EVALUATE1.yaml")
    parser.add_argument("--models", type=int, default=50)
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument("--limit", type=float, default=1.0)
    parser.add_argument("--points", type=Path, help="Named-column parameter table instead of random draws")
    parser.add_argument("--no-fisher", action="store_true",
                        help="Skip the derivative and contour stage")
    parser.add_argument("--output", type=Path,
                        help="New result directory below project chains/")
    parser.add_argument("--plot-only", type=Path,
                        help="Redraw a completed comparison directory")
    parser.add_argument("--fisher-sample-power", type=int, default=18,
                        help="log2 bounded-Fisher proposals per repetition")
    parser.add_argument("--resample-fisher", action="store_true",
                        help="With --plot-only, repeat bounded sampling")
    parser.add_argument("--advisory", action="store_true",
                        help="Report error sizes; fail only incomplete calculations")
    parser.add_argument("--local-points", type=Path,
                        default=None,
                        help="Replace the default bounded-Fisher local points")
    parser.add_argument("--mpi", action="store_true",
                        help="Distribute points and Fisher columns across MPI ranks")
    args = parser.parse_args()
    # Each rank owns its Cobaya model and native state. The shared test
    # distributes different jobs and lets rank zero write combined results.
    # Cobaya's own MPI remains disabled; this driver owns the work split.
    mpi_comm = None
    if args.mpi:
        from mpi4py import MPI
        mpi_comm = MPI.COMM_WORLD
    elif int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1")) > 1:
        parser.error("Use --mpi with mpirun to distribute work without duplicate writes.")
    if args.models < 1 or args.limit <= 0:
        parser.error("models and limit must be positive")
    points_file = args.points.resolve() if args.points is not None else None
    if args.plot_only is not None:
        if mpi_comm is not None and mpi_comm.Get_rank() != 0:
            sys.exit(0)
        output = args.plot_only.resolve()
        if args.resample_fisher:
            sample_matter_emulator_fisher(
                output=output / "fisher",
                sample_power=args.fisher_sample_power,
            )
        local = output / "local"
        plot_matter_emulator_fisher(
            output=output / "fisher",
            points_file=output / "points.txt",
            distances_file=output / "distances.txt",
            project_name=project.name,
            local_points_file=local / "points.txt" if local.is_dir() else None,
            local_distances_file=local / "distances.txt" if local.is_dir() else None,
        )
        sys.exit(0)
    output_dir = args.output.resolve() if args.output is not None else None
    local_points = args.local_points.resolve() if args.local_points is not None else None
    os.chdir(root)
    # The cold-field fit is validated for LCDM. Keep the numbered source
    # example intact and save the exact fixed-dark-energy test input in
    # chains/. Its amplitude cut is redundant over this capped prior box:
    # even As=max and all lens biases=max stay below its boundary.
    from cobaya.input import update_info
    from cobaya.yaml import yaml_load_file, yaml_dump
    info = yaml_load_file(str(project / args.configuration))
    expanded = update_info(info)
    expected_cut = "lambda As_1e9, DES_B1_1, DES_B1_2, DES_B1_3: 0.0 if (As_1e9/2.19)*max((DES_B1_1/1.42)**2, (DES_B1_2/1.66)**2, (DES_B1_3/1.7)**2) <= 8 else -np.inf"
    if info.get("prior", {}) != {"amplitude_cut": expected_cut}:
        raise ValueError("This cluster test requires example 1's documented amplitude cut.")
    maximum = expanded["params"]["As_1e9"]["prior"]["max"] / 2.19
    for bias, reference in (("DES_B1_1", 1.42), ("DES_B1_2", 1.66), ("DES_B1_3", 1.7)):
        bounds = expanded["params"][bias]["prior"]
        largest_bias = max(abs(bounds["min"]), abs(bounds["max"]))
        if maximum * (largest_bias / reference)**2 > 8:
            raise ValueError("The amplitude cut is active; its Fisher treatment needs review.")
    info.pop("prior")
    for parameter in ("w", "w0pwa"):
        info["params"][parameter] = {"value": -1.0}
        info["sampler"]["evaluate"]["override"].pop(parameter, None)
    if output_dir is None:
        output_dir = project / "chains/matter_emulator_prior" / Path(args.configuration).stem / "with_fisher"
    output_dir.mkdir(parents=True, exist_ok=True)
    configuration = output_dir / "lcdm_configuration.yaml"
    configuration.write_text(yaml_dump(info))
    args.configuration = str(configuration)
    failed = run_matter_emulator_prior_check(
        project=project,
        mpi_comm=mpi_comm,
        configuration=args.configuration,
        get_precision=ci.get_inv_cov_masked_cluster,
        candidate_extra_args={"cold_power": "eisenstein_hu", "cold_neutrino_species": 1},
        n_models=args.models,
        seed=args.seed,
        limit=args.limit,
        fisher=not args.no_fisher,
        output_dir=output_dir,
        fisher_sample_power=args.fisher_sample_power,
        points_file=points_file,
        advisory=args.advisory,
        local_points_file=local_points,
    )
    sys.exit(1 if failed else 0)
