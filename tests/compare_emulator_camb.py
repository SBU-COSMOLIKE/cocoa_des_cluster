"""Compare the candidate matter emulator with CAMB on the fixed CPL benchmark.

Run from activated Cocoa (one project at a time):
    python projects/des_cluster/tests/compare_emulator_camb.py
    mpirun -n 2 --bind-to none python -m mpi4py \
      projects/des_cluster/tests/compare_emulator_camb.py --mpi

MPI assigns independent parameter points or Fisher columns to each rank;
OpenMP accelerates each rank's model evaluations. Keep BLAS at one thread.

The shared Cocoa test replays the saved broad and focused parameter
tables with one massive neutrino at 0.06 eV. Both routes use the same
nuisance values and plain mead2020. The EMUL2 candidate uses emulated
distances and spectra.

Use EMULATOR_CODE_DIR and EMULATOR_DATA_DIR to select candidate checkouts.
COCOA_TEST_INTERFACE_DIR optionally selects an independently built native
interface. Results, including failed points, go under chains/.

For a custom configuration, the cluster test verifies that the amplitude
cut is redundant over the full prior box before removing it.

The normal command compares fixed broad and focused cosmologies and uses a
saved Fisher reference for the contour when available. The vector test is
independent of the Fisher diagnostic. Run check_fom_convergence.py
separately to assess CAMB Fisher numerical settings.
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
)


def main(*, fisher=False):
    description = "Screen the CAMB Fisher FoM with one fixed stencil spacing." if fisher else __doc__
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--configuration", default=None)
    parser.add_argument("--models", type=int, default=50)
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument("--limit", type=float, default=1.0)
    parser.add_argument("--points", type=Path, help="Named-column parameter table instead of random draws")
    parser.set_defaults(no_fisher=False, fisher_reference=None, plot_only=None,
                        validate_fisher_steps=False, save_fisher_contours=False)
    if fisher:
        parser.add_argument("--validate-fisher-steps", action="store_true",
                            help="Explicitly audit full h/h2 and higher-accuracy grids")
        parser.add_argument("--save-fisher-contours", action="store_true",
                            help="Also integrate bounded contours for future plots")
    else:
        parser.add_argument("--no-fisher", action="store_true",
                            help="Omit the saved Fisher contour")
        parser.add_argument("--fisher-reference", type=Path,
                            help="Use a different saved Fisher reference directory")
    parser.add_argument("--output", type=Path,
                        help="New result directory below project chains/")
    if not fisher:
        parser.add_argument("--plot-only", type=Path,
                            help="Redraw a completed comparison directory")
    parser.add_argument("--fisher-sample-power", type=int, default=18,
                        help="log2 bounded-Fisher proposals per repetition")
    parser.add_argument("--advisory", action="store_true",
                        help="Report error sizes; fail only incomplete calculations")
    parser.add_argument("--local-points", type=Path,
                        default=None,
                        help="Replace the default bounded-Fisher local points")
    parser.add_argument("--mpi", action="store_true",
                        help="Distribute points and Fisher columns across MPI ranks")
    args = parser.parse_args()
    benchmark = project / "tests/fisher_reference"
    if not fisher:
        benchmark /= "w0wa"
    benchmark_yaml = benchmark / "benchmark.yaml"
    if args.configuration is None:
        source = project / "EXAMPLE_EVALUATE1.yaml" if fisher else benchmark_yaml
        args.configuration = str(source)
    standard_configuration = (project / args.configuration).resolve() == benchmark_yaml
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
    official = (standard_configuration and args.models == 50
                and args.seed == 137 and points_file is None)
    fisher_reference = args.fisher_reference
    if fisher_reference is not None:
        fisher_reference = fisher_reference.resolve()
    if official and (not fisher or args.save_fisher_contours):
        points_file = benchmark / "prior_50.txt"
        if local_points is None:
            local_points = benchmark / "focused_25.txt"
        for required_table in (points_file, local_points):
            if not required_table.is_file():
                parser.error(f"Required benchmark table is missing: {required_table}")
    if standard_configuration and fisher_reference is None:
        fisher_reference = benchmark
    os.chdir(root)
    configuration = project / args.configuration
    if not standard_configuration:
        # The amplitude cut is redundant over this capped prior box:
        # even As=max and all lens biases=max stay below its boundary.
        from cobaya.input import update_info
        from cobaya.yaml import yaml_load_file, yaml_dump
        info = yaml_load_file(str(project / args.configuration))
        expanded = update_info(info)
        bias_references = (
            ("DES_B1_1", 1.42),
            ("DES_B1_2", 1.66),
            ("DES_B1_3", 1.7),
        )
        if fisher and "des_cluster.combo_6x2pt_N" in info["likelihood"]:
            bias_references += (
                ("DES_B1_4", 1.62),
                ("DES_B1_5", 1.78),
                ("DES_B1_6", 1.75),
            )
        bias_names = ", ".join(bias for bias, _ in bias_references)
        ratios = ", ".join(f"({bias}/{reference:g})**2"
                           for bias, reference in bias_references)
        expected_cut = (f"lambda As_1e9, {bias_names}: 0.0 if "
                        f"(As_1e9/2.19)*max({ratios}) <= 8 else -np.inf")
        priors = info.get("prior", {})
        if (priors.get("amplitude_cut") != expected_cut
                or set(priors) - {"amplitude_cut", "emulator_omegabh2_cap"}):
            raise ValueError("This cluster test requires the example's documented amplitude cut.")
        maximum = expanded["params"]["As_1e9"]["prior"]["max"] / 2.19
        for bias, reference in bias_references:
            bounds = expanded["params"][bias]["prior"]
            largest_bias = max(abs(bounds["min"]), abs(bounds["max"]))
            if maximum * (largest_bias / reference)**2 > 8:
                raise ValueError("The amplitude cut is active; its Fisher treatment needs review.")
        del priors["amplitude_cut"]
        if not priors:
            del info["prior"]
        if output_dir is None:
            kind = "fisher_calibration" if fisher else "comparison"
            output_dir = project / "chains/matter_emulator_prior" / Path(args.configuration).stem / kind
        filename = "fom_configuration.yaml" if fisher else "comparison_configuration.yaml"
        configuration = output_dir / filename
        if mpi_comm is None or mpi_comm.Get_rank() == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            configuration.write_text(yaml_dump(info))
        if mpi_comm is not None:
            mpi_comm.Barrier()
    failed = run_matter_emulator_prior_check(
        project=project,
        mpi_comm=mpi_comm,
        configuration=configuration,
        get_precision=ci.get_inv_cov_masked_cluster,
        candidate_extra_args={"cold_power": "eisenstein_hu", "cold_neutrino_species": 1},
        n_models=args.models,
        seed=args.seed,
        limit=args.limit,
        fisher=fisher,
        validate_fisher_steps=args.validate_fisher_steps,
        save_fisher_contours=args.save_fisher_contours,
        fisher_reference=None if args.no_fisher else fisher_reference,
        focused_models=25 if official or local_points is not None else 0,
        output_dir=output_dir,
        fisher_sample_power=args.fisher_sample_power,
        points_file=points_file,
        advisory=args.advisory,
        local_points_file=local_points,
    )
    sys.exit(1 if failed else 0)

if __name__ == "__main__":
    main()
