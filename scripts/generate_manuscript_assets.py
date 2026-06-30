#!/usr/bin/env python
"""
Audit manuscript-facing reproducibility outputs.

Generation is intentionally split by setting:
- `scripts/generate_scz_assets.py`
- `scripts/generate_inka_assets.py`
- `scripts/generate_simulation_assets.py`
- `scripts/run_fdr_sensitivity_analysis.py`
"""

import argparse

from manuscript_assets import audit_outputs, ensure_dirs


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true", help="Report which manuscript outputs are present.")
    return parser.parse_args()


def main():
    parse_args()
    ensure_dirs()
    audit_outputs()


if __name__ == "__main__":
    main()
