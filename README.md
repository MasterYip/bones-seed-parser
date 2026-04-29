# bones_seed_parser

Browse, filter, and batch-export [Bones-seed](https://bones-seed.com/) G1 motions to NPZ — no Isaac Lab required.

## Requirements

| Package | Notes |
|---|---|
| `pandas` | metadata loading |
| `torch` | tensor math |
| `pytorch-kinematics` | standalone FK (`pip install pytorch-kinematics`) |
| `tqdm` | batch progress bars |
| `gradio` | GUI only (`pip install gradio`) |

All packages are already present in the `env_isaaclab` conda environment.

A G1 URDF file is required for FK conversion. The converter and GUI accept the path as an argument.

## Setup

Run every command from the `RLTracker/` directory with `scripts/` on the Python path:

```bash
cd RLTracker/
export PYTHONPATH=scripts:$PYTHONPATH
```

Or prepend `PYTHONPATH=scripts` to individual commands as shown below.

---

## CLI

### 1. Filter metadata

Print or save a filtered subset of the 142 k-motion metadata table.

```bash
# Print results to stdout
PYTHONPATH=scripts python -m bones_seed_parser.cli filter \
    --bones-seed artifacts/bones-seed \
    --name "walk|run" \
    --no-mirrors \
    --max-frames 600

# Save to CSV for use with 'convert'
PYTHONPATH=scripts python -m bones_seed_parser.cli filter \
    --bones-seed artifacts/bones-seed \
    --name "^walk" \
    --category "Basic Locomotion Neutral" \
    --no-mirrors \
    --min-frames 120 \
    --max-frames 3600 \
    --gender M \
    --out /tmp/walk_list.csv
```

**All filter flags**

| Flag | Description |
|---|---|
| `--name REGEX` | Case-insensitive regex across name and description columns |
| `--package PKG [PKG …]` | Package names (e.g. `Locomotion Dances`) |
| `--category CAT [CAT …]` | Category names |
| `--no-mirrors` | Exclude mirrored takes |
| `--min-frames N` / `--max-frames N` | Frame count bounds |
| `--min-duration SEC` / `--max-duration SEC` | Duration bounds in seconds (source is 120 fps) |
| `--movement-type TYPE [TYPE …]` | `content_type_of_movement` values |
| `--body-position POS [POS …]` | `content_body_position` values |
| `--gender M\|F` | Actor gender |
| `--height H [H …]` | Actor height category (e.g. `S M T`) |
| `--min-age N` / `--max-age N` | Actor age range |
| `--out CSV` | Save results to this file instead of printing |

---

### 2. Convert a single file

```bash
PYTHONPATH=scripts python -m bones_seed_parser.cli convert-one \
    --csv  artifacts/bones-seed/g1/csv/210531/jump_and_land_heavy_001__A001.csv \
    --output /tmp/jump.npz \
    --urdf  /path/to/g1.urdf \
    --fps   50 \
    --device cpu
```

Output NPZ keys match `csv_to_npz.py`:

| Key | Shape | Unit |
|---|---|---|
| `fps` | `[1]` | — |
| `joint_pos` | `[T, 29]` | rad, MUJOCO order |
| `joint_vel` | `[T, 29]` | rad/s |
| `body_pos_w` | `[T, 30, 3]` | m |
| `body_quat_w` | `[T, 30, 4]` | wxyz |
| `body_lin_vel_w` | `[T, 30, 3]` | m/s |
| `body_ang_vel_w` | `[T, 30, 3]` | rad/s |

---

### 3. Batch convert

Use the CSV produced by `filter --out` (or provide inline filter flags):

```bash
# Using a pre-filtered list
PYTHONPATH=scripts python -m bones_seed_parser.cli convert \
    --bones-seed  artifacts/bones-seed \
    --motions     /tmp/walk_list.csv \
    --output-dir  artifacts/npz_export \
    --urdf        /path/to/g1.urdf \
    --fps         50 \
    --device      cpu \
    --batch-size  512

# Inline filter (no pre-filter CSV needed)
PYTHONPATH=scripts python -m bones_seed_parser.cli convert \
    --bones-seed artifacts/bones-seed \
    --name "^walk" --no-mirrors --max-frames 600 \
    --output-dir artifacts/npz_export \
    --urdf       /path/to/g1.urdf
```

Output NPZ files mirror the CSV sub-path structure:
`artifacts/bones-seed/g1/csv/210531/walk_001.csv` →
`artifacts/npz_export/g1/csv/210531/walk_001.npz`

---

### 4. Launch the web GUI

```bash
PYTHONPATH=scripts python -m bones_seed_parser.cli gui \
    --bones-seed artifacts/bones-seed \
    --port 7860
```

Open `http://localhost:7860` in a browser.

**GUI layout**

```
┌─────────────────────┬──────────────────────────────────────────┐
│  Filters            │  Results table                           │
│  ─────────────────  │  (move_name, category, duration, …)      │
│  Name search        │                                          │
│  Package checkboxes │  Selected move names (one per line       │
│  Category checkboxes│  or "ALL" to export full filter result)  │
│  Duration slider    │                                          │
│  Mirror toggle      ├──────────────────────────────────────────┤
│  Movement type      │  Export to NPZ                           │
│  Body position      │  URDF path / output dir / FPS / device   │
│  Actor gender       │  [Export Selected → NPZ]                 │
│  Actor height       │  live export log                         │
│  [Apply Filter]     │                                          │
└─────────────────────┴──────────────────────────────────────────┘
```

---

## Python API

```python
import sys
sys.path.insert(0, "scripts")   # from RLTracker/

from bones_seed_parser import BonesSeedParser, BonesCSVConverter, launch_gui

# ── 1. Filter ─────────────────────────────────────────────────
parser = BonesSeedParser("artifacts/bones-seed")
df = parser.filter(
    name_pattern  = "walk",
    exclude_mirrors = True,
    min_frames    = 120,
    max_frames    = 3600,
    actor_gender  = "M",
)
print(f"{len(df)} motions matched")

# Resolve CSV path for any row
csv_path = parser.get_csv_path(df.iloc[0])

# ── 2. Convert one file ────────────────────────────────────────
conv = BonesCSVConverter(
    urdf_path  = "/path/to/g1.urdf",
    output_fps = 50,
    device     = "cpu",
)
conv.convert_file(csv_path, "/tmp/output.npz")

# ── 3. Batch convert ───────────────────────────────────────────
def on_progress(done, total, name):
    print(f"  [{done}/{total}] {name}")

conv.convert_batch(
    rows_df    = df,
    output_dir = "artifacts/npz_export",
    base_path  = "artifacts/bones-seed",
    progress_cb = on_progress,
)

# ── 4. GUI ─────────────────────────────────────────────────────
launch_gui("artifacts/bones-seed", port=7860)
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'bones_seed_parser'`**  
Run from `RLTracker/` with `PYTHONPATH=scripts` prepended, or `sys.path.insert(0, "scripts")` in Python.

**`ModuleNotFoundError: No module named 'pytorch_kinematics'`**  
```bash
pip install pytorch-kinematics
```

**`URDF joint '…' not in MUJOCO_DOF_NAMES`**  
The URDF passed to `BonesCSVConverter` is not the G1 29-DOF model. Use the standard Unitree G1 URDF.

**`CSV not found`**  
`move_g1_path` in the metadata is relative to `bones_seed_base`. Ensure `--bones-seed` points to the dataset root containing the `g1/csv/` tree.

**Body positions look wrong after conversion**  
Profile the NPZ with the standalone profiler:
```bash
python test/read_motion_file_profile.py /tmp/output.npz
```
`body_pos_w` should be in the range ±2 m. Values in the hundreds of metres indicate a unit or rotation bug in the source CSV.
