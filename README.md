# bones-seed-parser

Browse, filter, and batch-export [Bones-seed](https://bones-seed.com/) G1 motions to NPZ — no Isaac Lab required.

> [!WARNING]
> This repository is still under development. Documentation is incomplete and the code may contain bugs.


## Installation

```bash
git clone https://github.com/MasterYip/bones-seed-parser.git
cd bones-seed-parser

# Minimal install (parser only)
pip install -e .

# With NPZ converter support
pip install -e ".[converter]"

# With Gradio GUI
pip install -e ".[gui]"

# Everything
pip install -e ".[all]"
```

All commands are run from the repo root. The package is importable as `bones_seed_parser`.

## Quick start

### GUI

```bash
# Launch the Gradio GUI
python -m bones_seed_parser.cli gui --bones-seed path/to/bones-seed
```

### CLI

```bash
# Filter metadata (print to stdout or save to CSV)
python -m bones_seed_parser.cli filter \
    --bones-seed path/to/bones-seed \
    --name "walk|run" --no-mirrors --max-frames 600

# Convert a single file
python -m bones_seed_parser.cli convert-one \
    --csv path/to/motion.csv \
    --output path/to/motion.npz \
    --urdf /path/to/g1.urdf

# Batch convert using a pre-filtered CSV
python -m bones_seed_parser.cli convert \
    --bones-seed path/to/bones-seed \
    --motions /tmp/walk_list.csv \
    --output-dir artifacts/npz_export \
    --urdf /path/to/g1.urdf
```

### Python API

```python
from bones_seed_parser import BonesSeedParser, BonesCSVConverter, launch_gui

# Filter metadata
parser = BonesSeedParser("path/to/bones-seed")
df = parser.filter(name_pattern="walk", exclude_mirrors=True)

# Convert to NPZ
conv = BonesCSVConverter(urdf_path="/path/to/g1.urdf", output_fps=50)
conv.convert_file("path/to/motion.csv", "/tmp/output.npz")

# Batch convert
conv.convert_batch(df, output_dir="artifacts/npz_export", base_path="path/to/bones-seed")

# Launch GUI
launch_gui("path/to/bones-seed", port=7860)
```

## Filter flags

| Flag | Description |
|---|---|
| `--name REGEX` | Case-insensitive regex across name and description |
| `--package PKG [PKG ...]` | Package names (e.g. `Locomotion Dances`) |
| `--category CAT [CAT ...]` | Category names |
| `--no-mirrors` | Exclude mirrored takes |
| `--min-frames N` / `--max-frames N` | Frame count bounds |
| `--min-duration SEC` / `--max-duration SEC` | Duration bounds (source is 120 fps) |
| `--movement-type TYPE [TYPE ...]` | `content_type_of_movement` values |
| `--body-position POS [POS ...]` | `content_body_position` values |
| `--gender M\|F` | Actor gender |
| `--height H [H ...]` | Actor height category (e.g. `S M T`) |
| `--min-age N` / `--max-age N` | Actor age range |
| `--out CSV` | Save results to file instead of printing |

## NPZ output keys

| Key | Shape | Unit |
|---|---|---|
| `fps` | `[1]` | — |
| `joint_pos` | `[T, 29]` | rad, MuJoCo order |
| `joint_vel` | `[T, 29]` | rad/s |
| `body_pos_w` | `[T, 30, 3]` | m |
| `body_quat_w` | `[T, 30, 4]` | wxyz |
| `body_lin_vel_w` | `[T, 30, 3]` | m/s |
| `body_ang_vel_w` | `[T, 30, 3]` | rad/s |

Output NPZ files mirror the CSV sub-path structure:
`bones-seed/g1/csv/210531/walk_001.csv` → `npz_export/g1/csv/210531/walk_001.npz`

## GUI

```bash
python -m bones_seed_parser.cli gui --bones-seed path/to/bones-seed --port 7860
```

Open `http://localhost:7860`. Pass `--theme github` or `--theme vscode` to change the colour theme (default: `default`). The theme can also be switched live from the UI.

| Flag | Default | Description |
|---|---|---|
| `--bones-seed PATH` | `artifacts/bones-seed` | Dataset root |
| `--metadata-version VER` | `latest` | Metadata version tag |
| `--host ADDR` | `0.0.0.0` | Bind address |
| `--port N` | `7860` | Port number |
| `--share` | off | Create a public Gradio share link |
| `--theme NAME` | `default` | `default`, `github`, or `vscode` |

## Requirements

| Package | Purpose |
|---|---|
| `pandas` | Metadata loading |
| `torch` | Tensor math |
| `pytorch-kinematics` | Standalone forward kinematics (`pip install pytorch-kinematics`) |
| `tqdm` | Batch progress bars |
| `gradio` | GUI only (`pip install gradio`) |

A G1 URDF file is required for FK conversion.

## File structure

```
bones_seed_parser/
├── __init__.py        # Package exports (BonesSeedParser, BonesCSVConverter, launch_gui)
├── cli.py             # Argparse entry-point (filter / convert / convert-one / gui)
├── parser.py          # Metadata loading & filtering
├── converter.py       # CSV → NPZ via pytorch-kinematics FK
├── gui.py             # Gradio web UI
├── themes/            # CSS theme files
│   ├── default.css
│   ├── github.css
│   └── vscode.css
├── viewer/            # Three.js 3D viewer (iframe in Gradio GUI)
│   ├── index.html
│   └── viewer.js
├── web/               # Standalone GitHub Pages static viewer
│   ├── index.html
│   ├── gen_web_data.py
│   └── themes/
└── models/
    └── g1_Zup_robot_01.fbx   # G1 skeleton for 3D preview
```

## Troubleshooting

**`URDF joint '…' not in MUJOCO_DOF_NAMES`**
The URDF passed to `BonesCSVConverter` is not the G1 29-DOF model. Use the standard Unitree G1 URDF.

**`CSV not found`**
`move_g1_path` in the metadata is relative to the dataset root. Ensure `--bones-seed` points to the directory containing the `g1/csv/` tree.

**Body positions look wrong after conversion**
`body_pos_w` should be in the range ±2 m. Values in the hundreds of metres indicate a unit or rotation bug in the source CSV.
