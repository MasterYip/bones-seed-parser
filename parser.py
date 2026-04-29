"""Bones-seed dataset metadata parser and motion filter.

Usage::

    from bones_seed_parser import BonesSeedParser

    parser = BonesSeedParser("/path/to/artifacts/bones-seed")
    df = parser.filter(name_pattern="walk", exclude_mirrors=True, min_frames=120)
    for _, row in df.iterrows():
        csv_path = parser.get_csv_path(row)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd


class BonesSeedParser:
    """Loads bones-seed metadata and provides filtering helpers.

    Args:
        base_path: Root directory of the bones-seed dataset, e.g.
            ``artifacts/bones-seed``.  Must contain a ``metadata/``
            sub-directory with at least one ``seed_metadata_v*.csv``.
        metadata_version: Explicit version tag like ``'004'``, or
            ``'latest'`` (default) to auto-select the highest version.
    """

    def __init__(
        self,
        base_path: str | Path,
        metadata_version: str = "latest",
    ) -> None:
        self.base_path = Path(base_path).resolve()
        self._csv_path = self._resolve_metadata(metadata_version)
        print(f"[BonesSeedParser] Loading metadata from {self._csv_path}")
        self.df = pd.read_csv(self._csv_path, low_memory=False)
        # Normalise boolean column: CSV stores 'True'/'False' strings
        if self.df["is_mirror"].dtype == object:
            self.df["is_mirror"] = self.df["is_mirror"].map(
                {"True": True, "False": False, True: True, False: False}
            )
        # Lazy-loaded temporal labels lookup
        self._temporal_labels: Optional[set] = None
        print(f"[BonesSeedParser] Loaded {len(self.df):,} motions × {len(self.df.columns)} columns")

    # ------------------------------------------------------------------
    # Temporal annotation helpers
    # ------------------------------------------------------------------

    def _load_temporal_labels(self) -> set[str]:
        """Lazy-load the set of move_names that have temporal annotations."""
        if self._temporal_labels is not None:
            return self._temporal_labels
        meta_dir = self.base_path / "metadata"
        jsonl_candidates = sorted(meta_dir.glob("seed_metadata_v*_temporal_labels.jsonl"))
        if not jsonl_candidates:
            print("[BonesSeedParser] WARNING: No temporal labels JSONL found.")
            self._temporal_labels = set()
            return self._temporal_labels
        jsonl_path = jsonl_candidates[-1]  # latest version
        print(f"[BonesSeedParser] Loading temporal labels from {jsonl_path.name} ...")
        labels: set[str] = set()
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    labels.add(entry["filename"])
                except (json.JSONDecodeError, KeyError):
                    continue
        self._temporal_labels = labels
        print(f"[BonesSeedParser] {len(labels):,} motions with temporal annotations loaded.")
        return self._temporal_labels

    def has_temporal_annotation(self, move_name: str) -> bool:
        """Check whether a single move_name has temporal annotations."""
        return move_name in self._load_temporal_labels()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_metadata(self, version: str) -> Path:
        meta_dir = self.base_path / "metadata"
        if not meta_dir.exists():
            raise FileNotFoundError(f"Metadata directory not found: {meta_dir}")
        csvs = sorted(meta_dir.glob("seed_metadata_v*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No seed_metadata_v*.csv files in {meta_dir}")
        if version == "latest":
            return csvs[-1]
        # match explicit version tag
        for p in csvs:
            if version in p.stem:
                return p
        raise FileNotFoundError(
            f"No metadata file matching version '{version}' in {meta_dir}. "
            f"Available: {[p.name for p in csvs]}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter(
        self,
        name_pattern: Optional[str] = None,
        packages: Optional[list[str]] = None,
        categories: Optional[list[str]] = None,
        exclude_mirrors: bool = False,
        min_frames: Optional[int] = None,
        max_frames: Optional[int] = None,
        movement_types: Optional[list[str]] = None,
        body_positions: Optional[list[str]] = None,
        actor_gender: Optional[str] = None,
        actor_height: Optional[list[str]] = None,
        min_age: Optional[float] = None,
        max_age: Optional[float] = None,
        min_duration_s: Optional[float] = None,
        max_duration_s: Optional[float] = None,
        has_temporal_annotation: Optional[bool] = None,
        fps: float = 120.0,
    ) -> pd.DataFrame:
        """Return a filtered view of the metadata DataFrame.

        Args:
            name_pattern: Case-insensitive regex / substring searched
                across ``move_name``, ``content_name``,
                ``content_natural_desc_1..4``, and
                ``content_short_description``.
            packages: Keep only rows whose ``package`` column value is
                in this list (exact match, case-insensitive).
            categories: Keep only rows whose ``category`` is in this list.
            exclude_mirrors: Drop mirrored motions (``is_mirror == True``).
            min_frames / max_frames: Frame count bounds (inclusive) using
                ``move_duration_frames``.
            movement_types: Filter on ``content_type_of_movement``.
            body_positions: Filter on ``content_body_position``.
            actor_gender: ``'M'`` or ``'F'`` (case-insensitive).
            actor_height: List of height categories, e.g. ``['S', 'M']``.
            min_age / max_age: Actor age bounds (years, inclusive).
            min_duration_s / max_duration_s: Duration bounds in seconds
                (converted to frames using *fps*).
            has_temporal_annotation: If ``True``, keep only motions with
                temporal labels in the JSONL.  If ``False``, keep only
                motions without temporal labels.  ``None`` (default)
                applies no temporal filtering.
            fps: Source FPS used when converting duration-in-seconds to
                frame counts (default 120).

        Returns:
            Filtered ``pd.DataFrame``.
        """
        mask = pd.Series(True, index=self.df.index)

        # --- name / description text search ---
        if name_pattern:
            text_cols = [
                "move_name", "content_name",
                "content_natural_desc_1", "content_natural_desc_2",
                "content_natural_desc_3", "content_natural_desc_4",
                "content_short_description", "content_short_description_2",
            ]
            pattern = re.compile(name_pattern, re.IGNORECASE)
            text_mask = pd.Series(False, index=self.df.index)
            for col in text_cols:
                if col in self.df.columns:
                    text_mask |= self.df[col].fillna("").str.contains(
                        pattern, na=False
                    )
            mask &= text_mask

        # --- package / category ---
        if packages:
            lower_pkgs = [p.lower() for p in packages]
            mask &= self.df["package"].str.lower().isin(lower_pkgs)
        if categories:
            lower_cats = [c.lower() for c in categories]
            mask &= self.df["category"].str.lower().isin(lower_cats)

        # --- mirror ---
        if exclude_mirrors:
            mask &= self.df["is_mirror"] != True  # noqa: E712

        # --- frame / duration ---
        if min_frames is not None:
            mask &= self.df["move_duration_frames"] >= min_frames
        if max_frames is not None:
            mask &= self.df["move_duration_frames"] <= max_frames
        if min_duration_s is not None:
            mask &= self.df["move_duration_frames"] >= min_duration_s * fps
        if max_duration_s is not None:
            mask &= self.df["move_duration_frames"] <= max_duration_s * fps

        # --- content annotation ---
        if movement_types:
            lower_mt = [m.lower() for m in movement_types]
            mask &= self.df["content_type_of_movement"].str.lower().isin(lower_mt)
        if body_positions:
            lower_bp = [b.lower() for b in body_positions]
            mask &= self.df["content_body_position"].str.lower().isin(lower_bp)

        # --- actor ---
        if actor_gender:
            mask &= self.df["actor_gender"].str.upper() == actor_gender.upper()
        if actor_height:
            lower_h = [h.upper() for h in actor_height]
            mask &= self.df["actor_height"].str.upper().isin(lower_h)
        if min_age is not None:
            mask &= self.df["actor_age_yr"] >= min_age
        if max_age is not None:
            mask &= self.df["actor_age_yr"] <= max_age

        # --- temporal annotation ---
        if has_temporal_annotation is not None:
            temporal_set = self._load_temporal_labels()
            in_temporal = self.df["move_name"].isin(temporal_set)
            if has_temporal_annotation:
                mask &= in_temporal
            else:
                mask &= ~in_temporal

        return self.df[mask].copy()

    def get_csv_path(self, row: pd.Series) -> Path:
        """Resolve the full path to a motion CSV from a metadata row.

        Args:
            row: A row from ``self.df`` (or the output of :meth:`filter`).

        Returns:
            Absolute ``Path`` to the CSV file.
        """
        return (self.base_path / row["move_g1_path"]).resolve()

    # ------------------------------------------------------------------
    # Unique-value helpers for GUI dropdowns
    # ------------------------------------------------------------------

    def list_packages(self) -> list[str]:
        return sorted(self.df["package"].dropna().unique().tolist())

    def list_categories(self) -> list[str]:
        return sorted(self.df["category"].dropna().unique().tolist())

    def list_movement_types(self) -> list[str]:
        return sorted(self.df["content_type_of_movement"].dropna().unique().tolist())

    def list_body_positions(self) -> list[str]:
        return sorted(self.df["content_body_position"].dropna().unique().tolist())

    def list_actor_heights(self) -> list[str]:
        return sorted(self.df["actor_height"].dropna().unique().tolist())

    def list_actor_genders(self) -> list[str]:
        return sorted(self.df["actor_gender"].dropna().unique().tolist())

    def get_unique(self, column: str) -> list:
        """Return sorted unique values for any column."""
        return sorted(self.df[column].dropna().unique().tolist())

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        return (
            f"BonesSeedParser(base={self.base_path}, "
            f"motions={len(self.df):,}, "
            f"metadata={self._csv_path.name})"
        )
