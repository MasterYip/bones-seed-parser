"""Bones-seed dataset parser, converter, and GUI.

Quick start::

    from bones_seed_parser import BonesSeedParser, BonesCSVConverter, launch_gui

    # Browse metadata
    parser = BonesSeedParser("artifacts/bones-seed")
    df = parser.filter(name_pattern="walk", exclude_mirrors=True)
    print(df[["move_name", "category", "move_duration_frames"]].head())

    # Batch-convert to NPZ (requires G1 URDF)
    conv = BonesCSVConverter(urdf_path="/path/to/g1.urdf", output_fps=50)
    conv.convert_batch(df, output_dir="artifacts/npz_export", base_path="artifacts/bones-seed")

    # Launch Gradio web UI
    launch_gui("artifacts/bones-seed")
"""

from .parser    import BonesSeedParser
from .converter import BonesCSVConverter
from .gui       import launch_gui, create_gui

__all__ = [
    "BonesSeedParser",
    "BonesCSVConverter",
    "launch_gui",
    "create_gui",
]
