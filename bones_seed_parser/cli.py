"""CLI entry point for bones_seed_parser.

Usage::

    # Launch Gradio GUI
    python -m bones_seed_parser.cli gui \\
        --bones-seed artifacts/bones-seed

    # Print filtered results to stdout (or save to CSV)
    python -m bones_seed_parser.cli filter \\
        --bones-seed artifacts/bones-seed \\
        --name "walk|run" --no-mirrors --out /tmp/walk_list.csv

    # Batch convert (requires G1 URDF)
    python -m bones_seed_parser.cli convert \\
        --bones-seed artifacts/bones-seed \\
        --motions /tmp/walk_list.csv \\
        --output-dir /tmp/npz_out \\
        --urdf /path/to/g1.urdf \\
        --fps 50

    # Convert a single CSV file
    python -m bones_seed_parser.cli convert-one \\
        --csv path/to/motion.csv \\
        --output path/to/motion.npz \\
        --urdf /path/to/g1.urdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Shared parent parser
# ---------------------------------------------------------------------------

def _add_bones_seed_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--bones-seed",
        default="artifacts/bones-seed",
        metavar="PATH",
        help="Root directory of the bones-seed dataset (default: artifacts/bones-seed).",
    )


def _add_metadata_version_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--metadata-version",
        default="latest",
        metavar="VER",
        help="Metadata version tag, e.g. '004', or 'latest' (default).",
    )


# ---------------------------------------------------------------------------
# Sub-command: gui
# ---------------------------------------------------------------------------

def cmd_gui(args: argparse.Namespace) -> None:
    from .gui import launch_gui

    launch_gui(
        bones_seed_base  = args.bones_seed,
        metadata_version = args.metadata_version,
        host             = args.host,
        port             = args.port,
        share            = args.share,
        theme            = args.theme,
    )


def _parser_gui(sub) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "gui",
        help="Launch the Gradio web UI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_bones_seed_arg(p)
    _add_metadata_version_arg(p)
    p.add_argument("--host",  default="0.0.0.0", help="Bind address.")
    p.add_argument("--port",  default=7860, type=int, help="Port number.")
    p.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    p.add_argument("--theme", default="default", choices=["default", "github", "vscode"],
                   help="UI colour theme (default: github).")
    p.set_defaults(func=cmd_gui)
    return p


# ---------------------------------------------------------------------------
# Sub-command: filter
# ---------------------------------------------------------------------------

def cmd_filter(args: argparse.Namespace) -> None:
    from .parser import BonesSeedParser

    parser = BonesSeedParser(args.bones_seed, args.metadata_version)
    df = parser.filter(
        name_pattern    = args.name,
        packages        = args.package or None,
        categories      = args.category or None,
        exclude_mirrors = args.no_mirrors,
        min_frames      = args.min_frames,
        max_frames      = args.max_frames,
        min_duration_s  = args.min_duration,
        max_duration_s  = args.max_duration,
        movement_types  = args.movement_type or None,
        body_positions  = args.body_position or None,
        actor_gender    = args.gender,
        actor_height    = args.height or None,
        min_age         = args.min_age,
        max_age         = args.max_age,
    )

    print(f"Filtered: {len(df):,} motions")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"Saved to {out}")
    else:
        # Print a concise table to stdout
        display_cols = [
            c for c in
            ["move_name", "package", "category", "move_duration_frames",
             "is_mirror", "actor_gender", "content_short_description"]
            if c in df.columns
        ]
        print(df[display_cols].to_string(index=False))


def _parser_filter(sub) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "filter",
        help="Filter metadata and print / save results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_bones_seed_arg(p)
    _add_metadata_version_arg(p)
    p.add_argument("--name",          default=None, metavar="REGEX",
                   help="Regex/substring across name and description columns.")
    p.add_argument("--package",       nargs="+", default=[], metavar="PKG",
                   help="One or more package names (case-insensitive).")
    p.add_argument("--category",      nargs="+", default=[], metavar="CAT",
                   help="One or more category names.")
    p.add_argument("--no-mirrors",    action="store_true",
                   help="Exclude mirrored motions.")
    p.add_argument("--min-frames",    type=int, default=None, metavar="N")
    p.add_argument("--max-frames",    type=int, default=None, metavar="N")
    p.add_argument("--min-duration",  type=float, default=None, metavar="SEC",
                   help="Minimum duration in seconds.")
    p.add_argument("--max-duration",  type=float, default=None, metavar="SEC",
                   help="Maximum duration in seconds.")
    p.add_argument("--movement-type", nargs="+", default=[], metavar="TYPE")
    p.add_argument("--body-position", nargs="+", default=[], metavar="POS")
    p.add_argument("--gender",        default=None, choices=["M", "F"])
    p.add_argument("--height",        nargs="+", default=[], metavar="H",
                   help="Actor height categories, e.g. S M T.")
    p.add_argument("--min-age",       type=float, default=None)
    p.add_argument("--max-age",       type=float, default=None)
    p.add_argument("--out",           default=None, metavar="CSV",
                   help="Save filtered metadata to this CSV path.")
    p.set_defaults(func=cmd_filter)
    return p


# ---------------------------------------------------------------------------
# Sub-command: convert (batch)
# ---------------------------------------------------------------------------

def cmd_convert(args: argparse.Namespace) -> None:
    import pandas as pd
    from .parser import BonesSeedParser
    from .converter import BonesCSVConverter

    # ---- Resolve which motions to convert --------------------------------
    if args.motions:
        motions_path = Path(args.motions)
        if not motions_path.exists():
            print(f"[ERROR] Motions CSV not found: {motions_path}", file=sys.stderr)
            sys.exit(1)
        df = pd.read_csv(motions_path, low_memory=False)
        if "move_g1_path" not in df.columns:
            parser = BonesSeedParser(args.bones_seed, args.metadata_version)
            names  = df["move_name"].tolist() if "move_name" in df.columns else df.iloc[:, 0].tolist()
            df     = parser.df[parser.df["move_name"].isin(names)]
    else:
        parser = BonesSeedParser(args.bones_seed, args.metadata_version)
        df = parser.filter(
            name_pattern    = args.name,
            packages        = args.package or None,
            categories      = args.category or None,
            exclude_mirrors = args.no_mirrors,
            min_frames      = args.min_frames,
            max_frames      = args.max_frames,
            movement_types  = args.movement_type or None,
        )

    # ---- Isaac Lab backend: print command and exit ------------------------
    if args.backend == "isaaclab":
        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix="_motions.csv", delete=False, prefix="bones_sel_"
        )
        df[["move_name", "move_g1_path"]].to_csv(tmp.name, index=False)
        tmp.close()
        flat_flag = "--flat " if args.flat else ""
        script = str(Path(__file__).parents[1] / "bones_seed_to_npz_isaaclab.py")
        print(f"\n[INFO] Wrote {len(df)} motion names to: {tmp.name}")
        print("\nRun this command with env_isaaclab:\n")
        print(
            f"  python {script} \\\n"
            f"    --bones-seed {args.bones_seed} \\\n"
            f"    --motions {tmp.name} \\\n"
            f"    --output-dir {args.output_dir} \\\n"
            f"    --fps {args.fps} {flat_flag}\\\n"
            f"    --max-envs {args.batch_size}"
        )
        return

    # ---- Torch backend ---------------------------------------------------
    if not args.urdf or not Path(args.urdf).exists():
        print(f"[ERROR] URDF file not found: {args.urdf}", file=sys.stderr)
        sys.exit(1)

    print(f"Converting {len(df):,} motions …")

    conv = BonesCSVConverter(
        urdf_path  = args.urdf,
        input_fps  = args.input_fps,
        output_fps = args.fps,
        device     = args.device,
        batch_size = args.batch_size,
    )
    conv.convert_batch(
        rows_df    = df,
        output_dir = args.output_dir,
        base_path  = args.bones_seed,
        flat       = args.flat,
    )


def _parser_convert(sub) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "convert",
        help="Batch convert CSV motions to NPZ.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_bones_seed_arg(p)
    _add_metadata_version_arg(p)
    _URDF_DEFAULT = "./source/rl_tracker/rl_tracker/assets/unitree_description/urdf/g1/main.urdf"
    p.add_argument("--urdf",        default=_URDF_DEFAULT, metavar="PATH",
                   help="Path to the G1 URDF file (default: %(default)s).")
    p.add_argument("--output-dir",  default="./artifacts/npz_export", metavar="DIR",
                   help="Root directory for NPZ output.")
    p.add_argument("--fps",         default=50, type=int, help="Output FPS.")
    p.add_argument("--input-fps",   default=120, type=int, help="Source FPS.")
    p.add_argument("--device",      default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--backend",     default="torch", choices=["torch", "isaaclab"],
                   help="'torch' uses standalone FK (cpu/cuda). "
                        "'isaaclab' prints the command to run "
                        "scripts/bones_seed_to_npz_isaaclab.py.")
    p.add_argument("--batch-size",  default=512, type=int, metavar="N",
                   help="FK batch size for torch; max-envs for isaaclab.")
    p.add_argument("--motions",     default=None, metavar="CSV",
                   help="CSV file of motions to convert (output of 'filter --out'). "
                        "If omitted, all motions passing inline filters are converted.")
    # Inline filter options (mirrors filter sub-command)
    p.add_argument("--name",          default=None, metavar="REGEX")
    p.add_argument("--package",       nargs="+", default=[])
    p.add_argument("--category",      nargs="+", default=[])
    p.add_argument("--no-mirrors",    action="store_true")
    p.add_argument("--min-frames",    type=int, default=None)
    p.add_argument("--max-frames",    type=int, default=None)
    p.add_argument("--movement-type", nargs="+", default=[])
    p.add_argument("--flat",          action="store_true",
                   help="Write {name}.npz directly under output-dir (default: {name}/motion.npz).")
    p.set_defaults(func=cmd_convert)
    return p


# ---------------------------------------------------------------------------
# Sub-command: convert-one
# ---------------------------------------------------------------------------

def cmd_convert_one(args: argparse.Namespace) -> None:
    from .converter import BonesCSVConverter

    if not Path(args.urdf).exists():
        print(f"[ERROR] URDF not found: {args.urdf}", file=sys.stderr)
        sys.exit(1)

    conv = BonesCSVConverter(
        urdf_path  = args.urdf,
        input_fps  = args.input_fps,
        output_fps = args.fps,
        device     = args.device,
        batch_size = args.batch_size,
    )
    conv.convert_file(args.csv, args.output)


def _parser_convert_one(sub) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "convert-one",
        help="Convert a single bones-seed CSV to NPZ.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _URDF_DEFAULT = "./source/rl_tracker/rl_tracker/assets/unitree_description/urdf/g1/main.urdf"
    p.add_argument("--csv",        required=True, metavar="PATH",
                   help="Input CSV file.")
    p.add_argument("--output",     required=True, metavar="PATH",
                   help="Output NPZ path.")
    p.add_argument("--urdf",       default=_URDF_DEFAULT, metavar="PATH",
                   help="Path to the G1 URDF file (default: %(default)s).")
    p.add_argument("--fps",        default=50, type=int)
    p.add_argument("--input-fps",  default=120, type=int)
    p.add_argument("--device",     default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--batch-size", default=512, type=int)
    p.set_defaults(func=cmd_convert_one)
    return p


# ---------------------------------------------------------------------------
# Root parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="bones_seed_parser",
        description="Bones-seed G1 dataset browser and NPZ exporter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = root.add_subparsers(dest="command")
    _parser_gui(sub)
    _parser_filter(sub)
    _parser_convert(sub)
    _parser_convert_one(sub)
    return root


def main(argv=None) -> None:
    parser = build_parser()
    args   = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
