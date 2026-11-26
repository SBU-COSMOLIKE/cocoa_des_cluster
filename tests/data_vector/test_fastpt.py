"""Compare the C and Python perturbation calculations in the optical TATT models.

The cosmic-shear, 3×2pt and 2×2pt checks use their saved DES Y3 models and
covariance. At each of the 30 intrinsic-alignment points the shared sweep
compares the two Python grids with the CFASTPT prediction at the same point.
"""

import os
from pathlib import Path
import sys

# OpenMP reads the thread count when the native library loads.
os.environ["OMP_NUM_THREADS"] = os.environ.get("COCOA_TEST_OMP_THREADS", "4")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cocoa_test_utils as u


TestCfastptVsFastptSweep = u.fastpt_sweep_suite(
    u,
    name="TestCfastptVsFastptSweep",
    cases=(
        ("test_x17_cfastpt_vs_fastpt_shear", 17, "example3", "Cosmic shear (DES Y3"),
        ("test_x18_cfastpt_vs_fastpt_3x2pt", 18, "example4", "3×2pt (DES Y3"),
        ("test_x19_cfastpt_vs_fastpt_2x2pt", 19, "example4_2x2pt", "2×2pt (DES Y3"),
    ),
)
