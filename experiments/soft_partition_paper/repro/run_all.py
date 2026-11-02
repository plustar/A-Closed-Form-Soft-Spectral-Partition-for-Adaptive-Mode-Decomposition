"""Run every reproducibility check and report pass/fail.

    python experiments/soft_partition_paper/repro/run_all.py
"""
from __future__ import annotations
import argparse, importlib, os, sys, time

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP  = os.path.dirname(os.path.dirname(_HERE))   # experiments/
_ROOT = os.path.dirname(_EXP)                     # repository root
for _p in (_ROOT, _EXP):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SCRIPTS = [
    ("s2_equivalence",   "Rem. IV.6, Rem. VI.5, Exp. A -- agreement with ADMM"),
    ("s5_algebraic",     "Thm. III.1, Cor. III.2, Rem. IV.2, Rem. VI.4"),
    ("s6_closed_form",   "Thm. IV.3, Rem. IV.4, Cor. IV.9, Rem. IV.10, Exp. S3"),
    ("s7_mmse",          "Thm. V.2, Rem. V.4, Sec. V-C, Exp. C"),
    ("s8_fwhm",          "Cor. V.5, Exp. S2"),
    ("s9_convergence",   "Prop. VI.1, Prop. VI.2, Rem. VI.3"),
    ("s10_reconstruction", "Rem. VII.1, App. G -- transforms and waveform recovery"),
    ("s11_boundary_energy", "Table IV, Exp. D, App. H"),
    ("unit_weighted_qp", "Cor. IV.7, Rem. IV.8, Thm. V.2, Rem. V.3"),
]


def main(only=None):
    run = [s for s in SCRIPTS if not only or s[0] in only]
    failures = []
    t0 = time.perf_counter()
    for name, desc in run:
        print("\n" + "#" * 78)
        print(f"# {name}  --  {desc}")
        print("#" * 78)
        t = time.perf_counter()
        try:
            mod = importlib.import_module(f"soft_partition_paper.repro.{name}")
            mod.main()
            print(f"\n[ok] {name} ({time.perf_counter() - t:.1f}s)")
        except Exception as exc:
            failures.append((name, exc))
            print(f"\n[FAIL] {name}: {type(exc).__name__}: {exc}")
    print("\n" + "=" * 78)
    print(f"{len(run) - len(failures)}/{len(run)} passed in {time.perf_counter() - t0:.1f}s")
    for name, exc in failures:
        print(f"  FAILED {name}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    sys.exit(main(ap.parse_args().only))
