"""Gradio web UI for bones-seed dataset browsing and NPZ export.

Launch::

    from bones_seed_parser import launch_gui
    launch_gui("/path/to/artifacts/bones-seed")

Or via CLI::

    python -m bones_seed_parser.cli gui --bones-seed artifacts/bones-seed
"""

from __future__ import annotations

import os
import threading
import traceback
from pathlib import Path
from typing import Optional

import pandas as pd

from .parser import BonesSeedParser

# Column subset shown in the results table
_DISPLAY_COLS = [
    "move_name",
    "package",
    "category",
    "move_duration_frames",
    "is_mirror",
    "actor_gender",
    "content_short_description",
    "move_g1_path",
]


def _safe_display(df: pd.DataFrame) -> pd.DataFrame:
    """Return a display-safe subset of columns."""
    cols = [c for c in _DISPLAY_COLS if c in df.columns]
    out = df[cols].copy()
    # Add duration in seconds (assume 120 fps)
    if "move_duration_frames" in out.columns:
        out.insert(
            out.columns.get_loc("move_duration_frames") + 1,
            "duration_s",
            (out["move_duration_frames"] / 120.0).round(2),
        )
    return out.reset_index(drop=True)


def create_gui(
    bones_seed_base: str,
    metadata_version: str = "latest",
) -> "gr.Blocks":
    """Build and return the Gradio ``Blocks`` application.

    Args:
        bones_seed_base:  Root directory of the bones-seed dataset.
        metadata_version: Metadata version tag or ``'latest'``.

    Returns:
        ``gr.Blocks`` (call ``.launch()`` to start).
    """
    try:
        import gradio as gr
    except ImportError as exc:
        raise ImportError(
            "gradio is required for the GUI. Install with:\n  pip install gradio"
        ) from exc

    parser = BonesSeedParser(bones_seed_base, metadata_version)

    # Pre-compute dropdown option lists
    pkgs      = parser.list_packages()
    cats      = parser.list_categories()
    mov_types = parser.list_movement_types()
    bod_pos   = parser.list_body_positions()
    heights   = parser.list_actor_heights()
    genders   = ["All"] + parser.list_actor_genders()

    max_frames = int(parser.df["move_duration_frames"].max())

    # ---- Shared mutable state ----
    _filtered_df: list[Optional[pd.DataFrame]] = [None]  # wrapped in list for closure

    # ------------------------------------------------------------------ #
    # Layout
    # ------------------------------------------------------------------ #
    with gr.Blocks(title="Bones-Seed Motion Browser", css="""
        #result_table { user-select: none; -webkit-user-select: none; }
    """) as demo:
        gr.Markdown(
            "## Bones-Seed Motion Browser\n"
            f"Dataset: `{bones_seed_base}` — "
            f"**{len(parser):,} motions** loaded"
        )

        # ──────────────────────────────────────────────────────────────
        # Row 1 — Filter panel + results table
        # ──────────────────────────────────────────────────────────────
        with gr.Row():

            # --- Filter sidebar ---
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### Filters")

                name_box = gr.Textbox(
                    label="Name / description search (regex OK)",
                    placeholder="e.g. walk, run, jump",
                )

                pkg_check = gr.CheckboxGroup(
                    choices=pkgs,
                    label="Package",
                    value=[],
                )
                cat_check = gr.CheckboxGroup(
                    choices=cats,
                    label="Category",
                    value=[],
                )
                mirror_radio = gr.Radio(
                    choices=["All", "Original only", "Mirror only"],
                    value="All",
                    label="Mirror",
                )
                min_dur_slider = gr.Slider(
                    minimum=0,
                    maximum=max_frames,
                    step=1,
                    value=0,
                    label="Min duration (frames)",
                )
                max_dur_slider = gr.Slider(
                    minimum=0,
                    maximum=max_frames,
                    step=1,
                    value=max_frames,
                    label="Max duration (frames)",
                )
                mov_type_dd = gr.Dropdown(
                    choices=mov_types,
                    multiselect=True,
                    label="Type of movement",
                    value=[],
                )
                bod_pos_dd = gr.Dropdown(
                    choices=bod_pos,
                    multiselect=True,
                    label="Body position",
                    value=[],
                )
                gender_radio = gr.Radio(
                    choices=genders,
                    value="All",
                    label="Actor gender",
                )
                height_check = gr.CheckboxGroup(
                    choices=heights,
                    label="Actor height",
                    value=[],
                )

                apply_btn = gr.Button("Apply Filter", variant="primary")

            # --- Results table ---
            with gr.Column(scale=3):
                result_info  = gr.Markdown("*Apply filter to see results.*")
                result_table = gr.Dataframe(
                    headers=_DISPLAY_COLS + ["duration_s"],
                    interactive=False,
                    wrap=True,
                    elem_id="result_table",
                )
                gr.Markdown(
                    "**Select motions to export** — paste `move_name` values "
                    "(one per line) into the box below, or type `ALL` to export "
                    "the entire filtered set."
                )
                selection_box = gr.Textbox(
                    label="Selected move names (one per line, or ALL)",
                    placeholder="jump_and_land_heavy_001__A001_M\nwalk_normal_001__A002",
                    lines=6,
                )

        # ──────────────────────────────────────────────────────────────
        # Row 2 — Export controls
        # ──────────────────────────────────────────────────────────────
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Export to NPZ")
                urdf_box = gr.Textbox(
                    label="G1 URDF path (required for FK)",
                    value="./source/rl_tracker/rl_tracker/assets/unitree_description/urdf/g1/main.urdf",
                )
                out_dir_box = gr.Textbox(
                    label="Output directory",
                    value="./artifacts/npz_export",
                )
                out_struct_radio = gr.Radio(
                    choices=["Named folder  (name/motion.npz)", "Flat  (name.npz)"],
                    value="Named folder  (name/motion.npz)",
                    label="Output structure",
                )
                out_fps_slider = gr.Slider(
                    minimum=10,
                    maximum=120,
                    step=1,
                    value=50,
                    label="Output FPS",
                )
                device_radio = gr.Radio(
                    choices=["torch (cpu)", "torch (cuda)", "isaaclab (batch)"],
                    value="torch (cpu)",
                    label="Conversion backend",
                )
                batch_size_num = gr.Number(
                    value=512,
                    label="FK batch size (frames per chunk)",
                    precision=0,
                )
                export_btn = gr.Button("Export Selected → NPZ", variant="primary")

            with gr.Column():
                export_log = gr.Textbox(
                    label="Export log",
                    lines=20,
                    interactive=False,
                )

        # ------------------------------------------------------------------ #
        # Callbacks
        # ------------------------------------------------------------------ #

        def apply_filter(
            name_pat, pkgs_sel, cats_sel, mirror_sel,
            min_dur, max_dur, mov_types_sel, bod_pos_sel,
            gender_sel, height_sel,
        ):
            exclude_mirrors = False
            mirror_only     = False
            if mirror_sel == "Original only":
                exclude_mirrors = True
            elif mirror_sel == "Mirror only":
                mirror_only = True

            df = parser.filter(
                name_pattern    = name_pat or None,
                packages        = pkgs_sel    or None,
                categories      = cats_sel    or None,
                exclude_mirrors = exclude_mirrors,
                min_frames      = int(min_dur) if int(min_dur) > 0 else None,
                max_frames      = int(max_dur) if int(max_dur) < max_frames else None,
                movement_types  = mov_types_sel or None,
                body_positions  = bod_pos_sel   or None,
                actor_gender    = None if gender_sel == "All" else gender_sel,
                actor_height    = height_sel or None,
            )

            if mirror_only:
                df = df[df["is_mirror"] == True]  # noqa: E712

            _filtered_df[0] = df
            info = f"**{len(df):,} motions** matched."
            return info, _safe_display(df)

        def do_export(
            selection_text, urdf, out_dir, out_struct, out_fps, backend, batch_sz,
        ):
            from .converter import BonesCSVConverter

            if _filtered_df[0] is None or len(_filtered_df[0]) == 0:
                yield "No filter results — apply a filter first."
                return

            df = _filtered_df[0]
            if selection_text.strip().upper() == "ALL" or not selection_text.strip():
                selected_df = df
            else:
                names = [n.strip() for n in selection_text.strip().splitlines() if n.strip()]
                selected_df = df[df["move_name"].isin(names)]
                if len(selected_df) == 0:
                    yield "No matching move names found in current filter results."
                    return

            # ---- Isaac Lab backend: generate CLI command, don't run directly ----
            if backend == "isaaclab (batch)":
                import tempfile
                # Write selected move names to a temp CSV so the script can read them
                tmp = tempfile.NamedTemporaryFile(
                    mode="w", suffix="_motions.csv", delete=False, prefix="bones_sel_"
                )
                selected_df[["move_name", "move_g1_path"]].to_csv(tmp.name, index=False)
                tmp.close()

                flat_flag = "--flat " if out_struct.startswith("Flat") else ""
                script = "scripts/bones_seed_to_npz_isaaclab.py"
                cmd = (
                    f"# Run from RLTracker/ with env_isaaclab:\n"
                    f"conda run -n env_isaaclab python {script} \\\n"
                    f"  --bones-seed {bones_seed_base} \\\n"
                    f"  --motions {tmp.name} \\\n"
                    f"  --output-dir {out_dir} \\\n"
                    f"  --fps {int(out_fps)} {flat_flag}\\\n"
                    f"  --max-envs {int(batch_sz)}"
                )
                lines = [
                    f"Selected {len(selected_df)} motions → saved to temp file: {tmp.name}",
                    "─" * 60,
                    "Run the following command in a terminal:",
                    "",
                    cmd,
                ]
                yield "\n".join(lines)
                return

            # ---- Torch backend (cpu / cuda) ----
            if not urdf or not Path(urdf).exists():
                yield f"URDF not found: {urdf}"
                return

            device = "cuda" if "cuda" in backend else "cpu"

            yield f"Initialising FK (URDF: {urdf}, device: {device}) …"

            try:
                conv = BonesCSVConverter(
                    urdf_path  = urdf,
                    output_fps = int(out_fps),
                    device     = device,
                    batch_size = int(batch_sz),
                )
            except Exception as exc:
                tb = traceback.format_exc()
                yield f"[ERROR] Failed to initialise converter (URDF: {urdf}):\n{exc}\n\n{tb}"
                return

            log_lines: list[str] = [
                f"Exporting {len(selected_df)} motions → {out_dir}",
                f"Output FPS: {out_fps}, backend: {backend}, structure: {out_struct}",
                "─" * 60,
            ]
            yield "\n".join(log_lines)

            flat = out_struct.startswith("Flat")
            done = 0
            total = len(selected_df)
            for _, row in selected_df.iterrows():
                move_name = row["move_name"]
                csv_src   = (Path(bones_seed_base) / row["move_g1_path"]).resolve()

                if flat:
                    out_npz = Path(out_dir) / f"{move_name}.npz"
                else:
                    out_npz = Path(out_dir) / move_name / "motion.npz"

                if not csv_src.exists():
                    log_lines.append(f"[SKIP] {move_name} — CSV not found: {csv_src}")
                    yield "\n".join(log_lines)
                    continue

                try:
                    conv.convert_file(csv_src, out_npz)
                    done += 1
                    log_lines.append(f"[{done}/{total}] OK  {move_name}")
                except Exception as exc:
                    tb = traceback.format_exc()
                    log_lines.append(
                        f"[{done}/{total}] ERR {move_name}: {exc}\n"
                        + "\n".join("  " + l for l in tb.splitlines())
                    )

                yield "\n".join(log_lines)

            log_lines.append("─" * 60)
            log_lines.append(f"Done: {done}/{total} motions exported to {out_dir}")
            yield "\n".join(log_lines)

        # Wire up
        filter_inputs = [
            name_box, pkg_check, cat_check, mirror_radio,
            min_dur_slider, max_dur_slider, mov_type_dd, bod_pos_dd,
            gender_radio, height_check,
        ]
        apply_btn.click(
            fn=apply_filter,
            inputs=filter_inputs,
            outputs=[result_info, result_table],
        )

        export_btn.click(
            fn=do_export,
            inputs=[selection_box, urdf_box, out_dir_box, out_struct_radio,
                    out_fps_slider, device_radio, batch_size_num],
            outputs=export_log,
        )

    return demo


def launch_gui(
    bones_seed_base: str,
    metadata_version: str = "latest",
    host: str = "0.0.0.0",
    port: int = 7860,
    share: bool = False,
) -> None:
    """Build and launch the Gradio GUI.

    Args:
        bones_seed_base:  Root directory of the bones-seed dataset.
        metadata_version: Metadata version or ``'latest'``.
        host:             Bind address.
        port:             Port number.
        share:            Create a public Gradio share link.
    """
    demo = create_gui(bones_seed_base, metadata_version)
    print(f"[BonesSeedGUI] Launching on http://{host}:{port}")
    demo.launch(server_name=host, server_port=port, share=share)
