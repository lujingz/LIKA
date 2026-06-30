#!/usr/bin/env python
"""
Generate manuscript assets for the INKA cell-line setting.

Default output is limited to files needed for manuscript reproducibility:
supplementary tables S2/S4 and main Figure 4A/B plus Figure 5.
"""

import argparse

from manuscript_assets import (
    FIGURE_DIR,
    INKA_RANKING_TABLE,
    INKA_SUBSTRATE_TABLE,
    ROOT,
    ensure_dirs,
    generate_inka_figures,
    write_kinase_ranking_table,
    write_substrate_statistics,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--use-existing-ranking",
        action="store_true",
        help="Use the committed S2 kinase ranking table instead of recomputing LIKA/KSEA rankings.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dirs()

    outputs = [
        write_substrate_statistics(
            ROOT / "data" / "intensity_data_INKA.csv",
            INKA_SUBSTRATE_TABLE,
            log_transform=True,
            include_all_substrates=True,
        )
    ]
    if not args.use_existing_ranking:
        outputs.append(
            write_kinase_ranking_table(
                ROOT / "data" / "intensity_data_INKA.csv",
                INKA_RANKING_TABLE,
                log_transform=True,
            )
        )
    outputs.extend(generate_inka_figures())

    for path in outputs:
        print(f"Wrote {path.relative_to(ROOT)}")
    print(f"INKA manuscript figures are in {FIGURE_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
