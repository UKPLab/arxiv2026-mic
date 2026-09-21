"""One command entry point for the individual construction stages."""

import argparse
import importlib
import sys

STAGES = {
    "reproduce": ("reproduce", "Recreate images from the released prompts and source URLs"),
    "prepare": ("images", "Normalize downloaded TARA metadata"),
    "download": ("images", "Download source images"),
    "filter": ("images", "Screen feasible contextual edits"),
    "verify": ("images", "Verify visible elements with a vision model"),
    "propose": ("images", "Assign types and generate editing prompts"),
    "edit": ("images", "Generate edited images"),
    "review": ("dataset", "Export review templates or assemble approved pairs"),
    "split": ("dataset", "Build entity and temporal splits"),
    "annotate": ("dataset", "Add optional teacher reasoning to reviewed data"),
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Stages:\n" + "\n".join(f"  {name:10} {description}" for name, (_, description) in STAGES.items())
        + "\n\nUse: python -m data_construction STAGE --help",
    )
    parser.add_argument("stage", choices=STAGES, help="Stage to run")
    if not argv:
        parser.print_help()
        return
    args = parser.parse_args(argv[:1])
    module, _ = STAGES[args.stage]
    entrypoint = getattr(importlib.import_module(f"data_construction.{module}"), f"{args.stage}_main")
    entrypoint(argv[1:])


if __name__ == "__main__":
    main()
