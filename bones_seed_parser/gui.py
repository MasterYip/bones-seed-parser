"""Gradio web UI for bones-seed dataset browsing and NPZ export.

Launch::

    from bones_seed_parser import launch_gui
    launch_gui("/path/to/artifacts/bones-seed")

Or via CLI::

    python -m bones_seed_parser.cli gui --bones-seed artifacts/bones-seed
"""

import json
import os
import socket
import threading
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional

import pandas as pd

from .parser import BonesSeedParser

# ---------------------------------------------------------------------------
# _ViewerServer — background HTTP side-channel for Three.js motion viewer
# ---------------------------------------------------------------------------

_VIEWER_DIR = Path(__file__).resolve().parent / "viewer"


def _find_free_port(start: int = 7861) -> int:
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found in range {start}–{start + 100}")


class _ViewerServer:
    """Background HTTP server that serves FBX model + motion CSV to the Three.js viewer."""

    def __init__(self, fbx_path: Path) -> None:
        self._fbx_path = Path(fbx_path)
        self._lock = threading.Lock()
        self._version: int = 0
        self._csv: str = ""
        self._name: str = ""

        self.port = _find_free_port()
        outer = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:  # suppress access log
                pass

            def _send_cors(self) -> None:
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "*")

            def do_OPTIONS(self) -> None:  # CORS pre-flight
                self.send_response(204)
                self._send_cors()
                self.end_headers()

            def do_GET(self) -> None:
                path = self.path.split("?")[0]
                if path == "/g1.fbx":
                    if not outer._fbx_path.exists():
                        self.send_response(404)
                        self.end_headers()
                        return
                    data = outer._fbx_path.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(len(data)))
                    self._send_cors()
                    self.end_headers()
                    self.wfile.write(data)

                elif path == "/anim-state":
                    with outer._lock:
                        payload = json.dumps({
                            "version": outer._version,
                            "name": outer._name,
                        }).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self._send_cors()
                    self.end_headers()
                    self.wfile.write(payload)

                elif path == "/csv":
                    with outer._lock:
                        csv_bytes = outer._csv.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/csv; charset=utf-8")
                    self.send_header("Content-Length", str(len(csv_bytes)))
                    self._send_cors()
                    self.end_headers()
                    self.wfile.write(csv_bytes)

                elif path in ("/viewer", "/viewer/", "/"):
                    data = (_VIEWER_DIR / "index.html").read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)

                elif path == "/viewer.js":
                    data = (_VIEWER_DIR / "viewer.js").read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/javascript; charset=utf-8")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)

                else:
                    self.send_response(404)
                    self.end_headers()

        httpd = HTTPServer(("127.0.0.1", self.port), _Handler)
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()

    def set_motion(self, csv_content: str, name: str) -> None:
        with self._lock:
            self._csv = csv_content
            self._name = name
            self._version += 1

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


# ---------------------------------------------------------------------------
# Viewer HTML — just an iframe pointing to the _ViewerServer's own page
# ---------------------------------------------------------------------------

def _build_viewer_html(server_url: str) -> str:
    """Return an iframe pointing to the standalone viewer served by _ViewerServer.

    The viewer page (viewer/index.html + viewer/viewer.js) runs at the same
    origin as the FBX and CSV endpoints, so no CORS or CSP issues.
    """
    return (
        f'<iframe src="{server_url}/viewer" '
        'style="width:100%;height:500px;border:none;border-radius:8px;'
        'background:#0e0e1a" '
        'loading="eager"></iframe>'
    )

# ---------------------------------------------------------------------------
# Column config + display helper
# ---------------------------------------------------------------------------

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

# G1 FBX model bundled inside this package (standalone — no seed-viewer dependency)
_FBX_PATH = Path(__file__).resolve().parent / "models" / "g1_Zup_robot_01.fbx"


def _safe_display(df: pd.DataFrame, selected: Optional[set] = None) -> pd.DataFrame:
    """Return a display-safe subset of columns, with a boolean *select* first column."""
    cols = [c for c in _DISPLAY_COLS if c in df.columns]
    out = df[cols].copy()
    if "move_duration_frames" in out.columns:
        out.insert(
            out.columns.get_loc("move_duration_frames") + 1,
            "duration_s",
            (out["move_duration_frames"] / 120.0).round(2),
        )
    sel_set = selected if selected is not None else set()
    out.insert(0, "select", out["move_name"].isin(sel_set))
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

_AUTO_REFRESH_THRESHOLD = 1000  # auto-refresh checkboxes when results ≤ this

_THEMES_DIR = Path(__file__).parent / "themes"
_VALID_THEMES = ("default", "github", "vscode")


def _load_theme_css(name: str) -> str:
    """Read ``themes/<name>.css`` next to this file. Falls back to ``default``."""
    css_file = _THEMES_DIR / f"{name}.css"
    if not css_file.is_file():
        css_file = _THEMES_DIR / "default.css"
    return css_file.read_text(encoding="utf-8")


# Pre-load all theme blobs once at import time
_CSS_BLOBS: dict[str, str] = {t: _load_theme_css(t) for t in _VALID_THEMES}


def _theme_html(name: str) -> str:
    """Wrap a theme's CSS in a hidden div so gr.HTML can inject it into the page."""
    css = _CSS_BLOBS.get(name, _CSS_BLOBS["default"])
    return f'<div style="display:none"><style id="bsp-theme-css">{css}</style></div>'


def create_gui(
    bones_seed_base: str,
    metadata_version: str = "latest",
    initial_theme: str = "default",
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

    # Pre-compute option lists
    pkgs      = parser.list_packages()
    cats      = parser.list_categories()
    mov_types = parser.list_movement_types()
    bod_pos   = parser.list_body_positions()
    heights   = parser.list_actor_heights()
    genders   = ["All"] + parser.list_actor_genders()
    max_frames = int(parser.df["move_duration_frames"].max())

    # Shared mutable state (wrapped in list to allow closure mutation)
    _filtered_df: list[Optional[pd.DataFrame]] = [None]
    _selected_names: list[set] = [set()]

    # Start motion-viewer side-channel server
    viewer_server = _ViewerServer(_FBX_PATH)
    viewer_html   = _build_viewer_html(viewer_server.url)

    # ------------------------------------------------------------------ #
    # Layout
    # ------------------------------------------------------------------ #
    with gr.Blocks(title="Bones-Seed Motion Browser") as demo:

        # Inject active theme CSS (swapped at runtime by the radio below)
        theme_css_html = gr.HTML(value=_theme_html(initial_theme))

        with gr.Row(equal_height=False):
            gr.Markdown(
                "## 🦴 Bones-Seed Motion Browser\n"
                f"Dataset: `{bones_seed_base}` — **{len(parser):,} motions** loaded"
            )
            # theme_radio = gr.Radio(
            #     choices=list(_VALID_THEMES),
            #     value=initial_theme,
            #     label="Theme",
            #     scale=0,
            #     min_width=220,
            # )

        with gr.Row(equal_height=False):

            # ── LEFT: filter sidebar ───────────────────────────────────
            with gr.Column(scale=1, min_width=260, elem_classes=["filter-sidebar"]):

                apply_btn = gr.Button("🔍 Apply Filter", variant="primary")

                with gr.Accordion("📋 Content", open=True):
                    name_box = gr.Textbox(
                        label="Name / description (regex OK)",
                        placeholder="walk, run, jump…",
                    )
                    pkg_check = gr.CheckboxGroup(choices=pkgs, label="Package", value=[])
                    cat_check = gr.CheckboxGroup(choices=cats, label="Category", value=[])
                    mirror_radio = gr.Radio(
                        choices=["All", "Original only", "Mirror only"],
                        value="All", label="Mirror",
                    )

                with gr.Accordion("⏱ Duration", open=False):
                    min_dur_slider = gr.Slider(0, max_frames, step=1, value=0,
                                               label="Min frames")
                    max_dur_slider = gr.Slider(0, max_frames, step=1, value=max_frames,
                                               label="Max frames")

                with gr.Accordion("🏃 Motion Type", open=False):
                    mov_type_dd = gr.Dropdown(choices=mov_types, multiselect=True,
                                              label="Type of movement", value=[])
                    bod_pos_dd  = gr.Dropdown(choices=bod_pos,   multiselect=True,
                                              label="Body position",   value=[])

                with gr.Accordion("👤 Actor", open=False):
                    gender_radio = gr.Radio(choices=genders, value="All",
                                            label="Gender")
                    height_check = gr.CheckboxGroup(choices=heights, label="Height",
                                                    value=[])

            # ── RIGHT: results + viewer ────────────────────────────────
            with gr.Column(scale=4):

                motion_info = gr.Markdown("*Click a row to preview the motion.*")
                gr.HTML(value=viewer_html, elem_id="viewer-panel")

                result_info = gr.Markdown("*Apply filter to see results.*")

                result_table = gr.Dataframe(
                    headers=["select"] + _DISPLAY_COLS + ["duration_s"],
                    interactive=False,
                    wrap=False,
                    elem_id="result_table",
                )

                with gr.Row():
                    select_all_btn   = gr.Button("☑ Select All",            elem_classes=["sel-btn"])
                    clear_btn        = gr.Button("✕ Clear",                  elem_classes=["sel-btn"])
                    refresh_sel_btn  = gr.Button("🔄 Refresh Selected Status", elem_classes=["refresh-btn"])

                selection_box = gr.Textbox(
                    label="Selected move names (one per line, or ALL)",
                    placeholder="jump_and_land_heavy_001__A001_M\nwalk_normal_001__A002",
                    lines=3,
                )

                with gr.Tabs():

                    # ── Tab: Export ────────────────────────────────────
                    with gr.Tab("📤 Export to NPZ"):
                        with gr.Row():
                            with gr.Column():
                                urdf_box = gr.Textbox(
                                    label="G1 URDF path (required for FK)",
                                    value="../../source/rl_tracker/rl_tracker/assets/unitree_description/urdf/g1/main.urdf",
                                )
                                out_dir_box = gr.Textbox(
                                    label="Output directory",
                                    value="./motion_export",
                                )
                                out_struct_radio = gr.Radio(
                                    choices=["Named folder  (name/motion.npz)", "Flat  (name.npz)"],
                                    value="Named folder  (name/motion.npz)",
                                    label="Output structure",
                                )
                                out_fps_slider = gr.Slider(10, 120, step=1, value=50,
                                                           label="Output FPS")
                                device_radio = gr.Radio(
                                    choices=["torch (cpu)", "torch (cuda)", "isaaclab (batch)"],
                                    value="torch (cpu)", label="Conversion backend",
                                )
                                batch_size_num = gr.Number(
                                    value=512, label="FK batch size (frames per chunk)",
                                    precision=0,
                                )
                                export_btn = gr.Button("Export Selected → NPZ",
                                                       variant="primary")
                            with gr.Column():
                                export_log = gr.Textbox(
                                    label="Export log", lines=20, interactive=False,
                                )

        # ------------------------------------------------------------------ #
        # Callbacks
        # ------------------------------------------------------------------ #

        def apply_filter(
            name_pat, pkgs_sel, cats_sel, mirror_sel,
            min_dur, max_dur, mov_types_sel, bod_pos_sel,
            gender_sel, height_sel,
        ):
            exclude_mirrors = mirror_sel == "Original only"
            mirror_only     = mirror_sel == "Mirror only"

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
            # Preserve existing selections — do NOT reset _selected_names
            info = f"**{len(df):,} motions** matched."
            # Auto-refresh checkboxes only for small result sets
            if len(df) <= _AUTO_REFRESH_THRESHOLD:
                return info, _safe_display(df, _selected_names[0])
            return info, _safe_display(df, set())

        def on_row_select(evt: gr.SelectData):
            """Column 0 (select) → toggle selection.  Any other column → preview motion."""
            if _filtered_df[0] is None:
                return gr.skip(), gr.skip(), "*No filter results.*"

            row_idx = evt.index[0]
            col_idx = evt.index[1]
            if row_idx >= len(_filtered_df[0]):
                return gr.skip(), gr.skip(), "*Row index out of range.*"

            row       = _filtered_df[0].iloc[row_idx]
            move_name = str(row.get("move_name", ""))

            if col_idx == 0:  # ── toggle selection (no table refresh) ──
                if move_name in _selected_names[0]:
                    _selected_names[0].discard(move_name)
                else:
                    _selected_names[0].add(move_name)
                new_sel = "\n".join(sorted(_selected_names[0]))
                return gr.skip(), new_sel, gr.skip()

            else:  # ── preview motion ──
                g1_path  = row.get("move_g1_path", "")
                csv_path = (Path(bones_seed_base) / g1_path).resolve() if g1_path else None
                if csv_path and csv_path.exists():
                    viewer_server.set_motion(csv_path.read_text(encoding="utf-8"), move_name)
                    preview_info = (
                        f"**{move_name}**  \n"
                        f"Package: `{row.get('package', '?')}`  |  "
                        f"Category: `{row.get('category', '?')}`  |  "
                        f"Duration: {row.get('move_duration_frames', '?')} frames "
                        f"({row.get('move_duration_frames', 0) / 120:.1f} s)  \n"
                        f"Gender: `{row.get('actor_gender', '?')}`  |  "
                        f"Mirror: `{row.get('is_mirror', '?')}`"
                    )
                else:
                    preview_info = f"**{move_name}** — CSV not found: `{g1_path}`"
                return gr.skip(), gr.skip(), preview_info

        def on_refresh_selected():
            """Manually refresh checkbox column to reflect current selection state."""
            if _filtered_df[0] is None:
                return gr.skip()
            return _safe_display(_filtered_df[0], _selected_names[0])

        def on_select_all():
            if _filtered_df[0] is None or len(_filtered_df[0]) == 0:
                return gr.skip(), ""
            _selected_names[0] = set(_filtered_df[0]["move_name"].tolist())
            new_sel = "\n".join(sorted(_selected_names[0]))
            return _safe_display(_filtered_df[0], _selected_names[0]), new_sel

        def on_clear():
            _selected_names[0] = set()
            if _filtered_df[0] is None:
                return gr.skip(), ""
            return _safe_display(_filtered_df[0], set()), ""

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

            # ---- Isaac Lab backend: generate CLI command ----
            if backend == "isaaclab (batch)":
                import tempfile
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
                yield "\n".join([
                    f"Selected {len(selected_df)} motions → {tmp.name}",
                    "─" * 60,
                    "Run in terminal:", "", cmd,
                ])
                return

            # ---- Torch backend ----
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
                yield f"[ERROR] {exc}\n\n{traceback.format_exc()}"
                return

            log_lines: list[str] = [
                f"Exporting {len(selected_df)} motions → {out_dir}",
                f"FPS: {out_fps}, backend: {backend}, structure: {out_struct}",
                "─" * 60,
            ]
            yield "\n".join(log_lines)

            flat = out_struct.startswith("Flat")
            done, total = 0, len(selected_df)
            for _, row in selected_df.iterrows():
                move_name = row["move_name"]
                csv_src   = (Path(bones_seed_base) / row["move_g1_path"]).resolve()
                out_npz   = (Path(out_dir) / f"{move_name}.npz") if flat else (
                             Path(out_dir) / move_name / "motion.npz")

                if not csv_src.exists():
                    log_lines.append(f"[SKIP] {move_name} — CSV not found")
                    yield "\n".join(log_lines)
                    continue
                try:
                    conv.convert_file(csv_src, out_npz)
                    done += 1
                    log_lines.append(f"[{done}/{total}] OK  {move_name}")
                except Exception as exc:
                    tb = "\n".join("  " + l for l in traceback.format_exc().splitlines())
                    log_lines.append(f"[{done}/{total}] ERR {move_name}: {exc}\n{tb}")
                yield "\n".join(log_lines)

            log_lines.append("─" * 60)
            log_lines.append(f"Done: {done}/{total} motions exported to {out_dir}")
            yield "\n".join(log_lines)

        # ── Wire up events ──────────────────────────────────────────────
        _filter_inputs = [
            name_box, pkg_check, cat_check, mirror_radio,
            min_dur_slider, max_dur_slider, mov_type_dd, bod_pos_dd,
            gender_radio, height_check,
        ]
        apply_btn.click(
            fn=apply_filter,
            inputs=_filter_inputs,
            outputs=[result_info, result_table],
        )
        result_table.select(
            fn=on_row_select,
            inputs=[],
            outputs=[result_table, selection_box, motion_info],
        )
        select_all_btn.click(fn=on_select_all, inputs=[], outputs=[result_table, selection_box])
        clear_btn.click(fn=on_clear, inputs=[], outputs=[result_table, selection_box])
        refresh_sel_btn.click(fn=on_refresh_selected, inputs=[], outputs=[result_table])
        export_btn.click(
            fn=do_export,
            inputs=[selection_box, urdf_box, out_dir_box, out_struct_radio,
                    out_fps_slider, device_radio, batch_size_num],
            outputs=export_log,
        )

        # theme_radio.change(
        #     fn=_theme_html,
        #     inputs=theme_radio,
        #     outputs=theme_css_html,
        #     queue=False,
        # )

    return demo


def launch_gui(
    bones_seed_base: str,
    metadata_version: str = "latest",
    host: str = "0.0.0.0",
    port: int = 7860,
    share: bool = False,
    theme: str = "default",
) -> None:
    """Build and launch the Gradio GUI.

    Args:
        bones_seed_base:  Root directory of the bones-seed dataset.
        metadata_version: Metadata version or ``'latest'``.
        host:             Bind address.
        port:             Port number.
        share:            Create a public Gradio share link.
        theme:            CSS theme to apply: ``'github'`` (default), ``'vscode'``, or ``'default'``.
    """
    demo = create_gui(bones_seed_base, metadata_version, initial_theme=theme)
    print(f"[BonesSeedGUI] Launching on http://{host}:{port} (theme={theme})")
    demo.launch(server_name=host, server_port=port, share=share)
