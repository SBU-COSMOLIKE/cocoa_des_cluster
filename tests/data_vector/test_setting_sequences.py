"""Setting sequences: settings changed in one process vs a fresh process.

A notebook changes cosmolike settings between evaluations without
restarting the kernel. Each sequence below changes one kind of setting
in one process and compares the result, bit for bit, with a process
that saw only the final settings, or expects the named error where the
change is refused. The shared sequences come from common_sequences and
the class from setting_sequence_suite (both in
cosmolike_core/cocoa_testing_suites.py, which documents every sequence
and the two worker legs); the project adds the sequences of its own
features below them.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/des_cluster/tests/data_vector/test_setting_sequences.py
"""

import os

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load, so
# this must run before any cobaya/cosmolike import in the process.
os.environ["OMP_NUM_THREADS"] = os.environ.get("COCOA_TEST_OMP_THREADS", "4")

import sys
import unittest

# The tests folder is not a package; put it on the import path so the
# shared harness resolves no matter where pytest was launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
import cocoa_test_utils as u

# The shared sequences on this project's examples: example 3 computes
# cosmic shear (4 source bins), example 4 has 5 lens bins.
SEQUENCES = u.common_sequences(
    "real", shear_example="example3", lens_example="example4", nsource=4,
    nlens=5, tatt=True, rebuild_example="example3")

# The cluster layout: a probe flag or the Y transform changed after
# init_data_cluster must stop with a named error, and so must a cluster
# evaluation after initial_setup alone (the binding also resets the
# cluster state).
CLUSTER = [
    {"name": "C1_cluster_probe_flip_after_init_data", "base": "example1",
     "steps": [[("call", {"name": "init_cluster_probes", "N": 1, "cs": 0,
                          "cc": 0, "cg": 0})]],
     "evaluate": "cluster", "expect": "fatal:call init_data_cluster again"},
    {"name": "C1_cluster_ytransform_flip_after_init_data", "base": "example1",
     "steps": [[("cluster_ytransform", {})]],
     "evaluate": "cluster", "expect": "fatal:call init_data_cluster again"},
    {"name": "C2_initial_setup_then_cluster_evaluation", "base": "example1",
     "steps": [[("call", {"name": "initial_setup"})]],
     "evaluate": "cluster", "expect": "fatal:init_data_cluster"},
]
SEQUENCES = CLUSTER + SEQUENCES

TestSettingSequences = u.setting_sequence_suite(
    u, name="TestSettingSequences", sequences=SEQUENCES)

# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
