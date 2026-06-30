#!/usr/bin/env python
"""
Generate manuscript assets for the SCZ/control setting.

Default output is limited to files needed for manuscript reproducibility:
supplementary tables S3/S5 and main Figure 6A/B.
"""

import argparse

from manuscript_assets import (
    FIGURE_DIR,
    ROOT,
    SCZ_RANKING_TABLE,
    SCZ_SUBSTRATE_TABLE,
    ensure_dirs,
    generate_scz_figures,
    write_kinase_ranking_table,
    write_substrate_statistics,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--use-existing-ranking",
        action="store_true",
        help="Use the committed S5 kinase ranking table instead of recomputing LIKA/KSEA rankings.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dirs()

    outputs = [
        write_substrate_statistics(
            ROOT / "data" / "residual_data_SCZ.csv",
            SCZ_SUBSTRATE_TABLE,
            log_transform=False,
            include_all_substrates=False,
        )
    ]
    if not args.use_existing_ranking:
        outputs.append(
            write_kinase_ranking_table(
                ROOT / "data" / "residual_data_SCZ.csv",
                SCZ_RANKING_TABLE,
                log_transform=False,
            )
        )
    outputs.extend(generate_scz_figures())

    for path in outputs:
        print(f"Wrote {path.relative_to(ROOT)}")
    print(f"SCZ/control manuscript figures are in {FIGURE_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
