"""Standalone batch converter: bones-seed G1 CSV → NPZ via pytorch-kinematics FK.

No Isaac Lab or Pinocchio required — only ``torch`` and ``pytorch-kinematics``.

Usage::

    from bones_seed_parser import BonesCSVConverter

    conv = BonesCSVConverter(urdf_path="path/to/g1.urdf", output_fps=50)
    conv.convert_file("g1/csv/210531/jump.csv", "output/jump.npz")

The NPZ format matches the output of ``csv_to_npz.py``:
    fps              – [1]        scalar
    joint_pos        – [T, 29]    radians, MUJOCO order
    joint_vel        – [T, 29]    rad/s,   MUJOCO order
    body_pos_w       – [T, 30, 3] metres,  ISAACLAB body order
    body_quat_w      – [T, 30, 4] wxyz,    ISAACLAB body order
    body_lin_vel_w   – [T, 30, 3] m/s,     ISAACLAB body order
    body_ang_vel_w   – [T, 30, 3] rad/s,   ISAACLAB body order
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# G1 joint / body names (copied from g1_reindex.py to keep converter standalone)
# ---------------------------------------------------------------------------

MUJOCO_DOF_NAMES = [
    "left_hip_pitch_joint",   "left_hip_roll_joint",   "left_hip_yaw_joint",
    "left_knee_joint",        "left_ankle_pitch_joint","left_ankle_roll_joint",
    "right_hip_pitch_joint",  "right_hip_roll_joint",  "right_hip_yaw_joint",
    "right_knee_joint",       "right_ankle_pitch_joint","right_ankle_roll_joint",
    "waist_yaw_joint",        "waist_roll_joint",       "waist_pitch_joint",
    "left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint",
    "left_elbow_joint",       "left_wrist_roll_joint",  "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint",
    "right_elbow_joint",      "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

ISAACLAB_BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link",  "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",   "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",       "right_knee_link",
    "left_shoulder_pitch_link", "right_shoulder_pitch_link",
    "left_ankle_pitch_link","right_ankle_pitch_link",
    "left_shoulder_roll_link",  "right_shoulder_roll_link",
    "left_ankle_roll_link", "right_ankle_roll_link",
    "left_shoulder_yaw_link","right_shoulder_yaw_link",
    "left_elbow_link",      "right_elbow_link",
    "left_wrist_roll_link", "right_wrist_roll_link",
    "left_wrist_pitch_link","right_wrist_pitch_link",
    "left_wrist_yaw_link",  "right_wrist_yaw_link",
]

# Isaac Lab DOF order: alphabetical from URDF (matches robot.data.joint_pos order)
ISAACLAB_DOF_NAMES = [
    "left_hip_pitch_joint",  "right_hip_pitch_joint",  "waist_yaw_joint",
    "left_hip_roll_joint",   "right_hip_roll_joint",   "waist_roll_joint",
    "left_hip_yaw_joint",    "right_hip_yaw_joint",    "waist_pitch_joint",
    "left_knee_joint",       "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint","right_ankle_pitch_joint",
    "left_shoulder_roll_joint",  "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint","right_shoulder_yaw_joint",
    "left_elbow_joint",      "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint","right_wrist_pitch_joint",
    "left_wrist_yaw_joint",  "right_wrist_yaw_joint",
]

# Precomputed permutation: ISAACLAB_DOF_NAMES[i] = MUJOCO_DOF_NAMES[_MUJOCO_TO_ISAACLAB[i]]
# Apply as:  q_isaaclab = q_mujoco[:, _MUJOCO_TO_ISAACLAB]
_MUJOCO_TO_ISAACLAB: list[int] = [
    MUJOCO_DOF_NAMES.index(name) for name in ISAACLAB_DOF_NAMES
]

N_BODIES = len(ISAACLAB_BODY_NAMES)   # 30
N_DOFS   = len(MUJOCO_DOF_NAMES)      # 29


# ---------------------------------------------------------------------------
# Pure-torch math helpers
# ---------------------------------------------------------------------------

def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Quaternion product (wxyz convention).  [..., 4] × [..., 4] → [..., 4]."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], dim=-1)


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """[..., 4] wxyz → conjugate [..., 4]."""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def _axis_angle_from_quat(q: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """[..., 4] wxyz → [..., 3] axis-angle."""
    # Ensure positive w hemisphere (shorter arc)
    q = torch.where(q[..., :1] < 0, -q, q)
    n  = q[..., 1:].norm(dim=-1, keepdim=True).clamp(min=eps)
    theta = 2.0 * torch.atan2(n, q[..., :1])
    return (q[..., 1:] / n) * theta


def _rotmat_to_quat_wxyz(R: torch.Tensor) -> torch.Tensor:
    """[..., 3, 3] rotation matrix → [..., 4] quaternion wxyz.

    Uses the Shepperd / numerically-stable method.
    """
    *batch, _, _ = R.shape
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]

    qw = torch.zeros(*batch, device=R.device, dtype=R.dtype)
    qx = torch.zeros_like(qw)
    qy = torch.zeros_like(qw)
    qz = torch.zeros_like(qw)

    # Case 1: trace > 0
    s1 = torch.sqrt((trace + 1.0).clamp(min=1e-10)) * 2.0  # 4w
    qw1 = 0.25 * s1
    qx1 = (R[..., 2, 1] - R[..., 1, 2]) / s1
    qy1 = (R[..., 0, 2] - R[..., 2, 0]) / s1
    qz1 = (R[..., 1, 0] - R[..., 0, 1]) / s1

    # Case 2: R[0,0] > R[1,1], R[0,0] > R[2,2]
    s2 = torch.sqrt((1.0 + R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2]).clamp(min=1e-10)) * 2.0
    qw2 = (R[..., 2, 1] - R[..., 1, 2]) / s2
    qx2 = 0.25 * s2
    qy2 = (R[..., 0, 1] + R[..., 1, 0]) / s2
    qz2 = (R[..., 0, 2] + R[..., 2, 0]) / s2

    # Case 3: R[1,1] > R[2,2]
    s3 = torch.sqrt((1.0 + R[..., 1, 1] - R[..., 0, 0] - R[..., 2, 2]).clamp(min=1e-10)) * 2.0
    qw3 = (R[..., 0, 2] - R[..., 2, 0]) / s3
    qx3 = (R[..., 0, 1] + R[..., 1, 0]) / s3
    qy3 = 0.25 * s3
    qz3 = (R[..., 1, 2] + R[..., 2, 1]) / s3

    # Case 4: default
    s4 = torch.sqrt((1.0 + R[..., 2, 2] - R[..., 0, 0] - R[..., 1, 1]).clamp(min=1e-10)) * 2.0
    qw4 = (R[..., 1, 0] - R[..., 0, 1]) / s4
    qx4 = (R[..., 0, 2] + R[..., 2, 0]) / s4
    qy4 = (R[..., 1, 2] + R[..., 2, 1]) / s4
    qz4 = 0.25 * s4

    c1 = trace > 0
    c2 = ~c1 & (R[..., 0, 0] > R[..., 1, 1]) & (R[..., 0, 0] > R[..., 2, 2])
    c3 = ~c1 & ~c2 & (R[..., 1, 1] > R[..., 2, 2])
    c4 = ~c1 & ~c2 & ~c3

    qw = torch.where(c1, qw1, torch.where(c2, qw2, torch.where(c3, qw3, qw4)))
    qx = torch.where(c1, qx1, torch.where(c2, qx2, torch.where(c3, qx3, qx4)))
    qy = torch.where(c1, qy1, torch.where(c2, qy2, torch.where(c3, qy3, qy4)))
    qz = torch.where(c1, qz1, torch.where(c2, qz2, torch.where(c3, qz3, qz4)))

    q = torch.stack([qw, qx, qy, qz], dim=-1)
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-10)


def _euler_zyx_deg_to_quat_wxyz(euler_deg: torch.Tensor) -> torch.Tensor:
    """Intrinsic ZYX Euler angles (degrees) → quaternion WXYZ.

    Columns are [rotateX, rotateY, rotateZ] applied in ZYX order, matching
    the bones-seed G1 CSV convention.
    """
    r  = torch.deg2rad(euler_deg)
    cx = torch.cos(r[..., 0] * 0.5);  sx = torch.sin(r[..., 0] * 0.5)
    cy = torch.cos(r[..., 1] * 0.5);  sy = torch.sin(r[..., 1] * 0.5)
    cz = torch.cos(r[..., 2] * 0.5);  sz = torch.sin(r[..., 2] * 0.5)
    w = cz*cy*cx + sz*sy*sx
    x = cz*cy*sx - sz*sy*cx
    y = cz*sy*cx + sz*cy*sx
    z = sz*cy*cx - cz*sy*sx
    return torch.stack([w, x, y, z], dim=-1)


def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """[..., 4] wxyz → [..., 3, 3] rotation matrix."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = torch.stack([
        torch.stack([1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)], dim=-1),
        torch.stack([  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)], dim=-1),
        torch.stack([  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)], dim=-1),
    ], dim=-2)
    return R


def _so3_derivative_per_body(
    body_quat: torch.Tensor, dt: float
) -> torch.Tensor:
    """Compute per-body angular velocity from quaternion sequence.

    Args:
        body_quat: [T, 30, 4] wxyz body quaternions in world frame.
        dt:        Time step in seconds.

    Returns:
        [T, 30, 3] angular velocities in world frame.
    """
    # [T-2, 30, 4]
    q_prev = body_quat[:-2]
    q_next = body_quat[2:]
    q_rel  = _quat_mul(q_next, _quat_conjugate(q_prev))   # [T-2, 30, 4]
    omega  = _axis_angle_from_quat(q_rel) / (2.0 * dt)    # [T-2, 30, 3]
    # Pad first and last sample
    omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)  # [T, 30, 3]
    return omega


def _slerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Quaternion slerp between two [N, 4] arrays at per-sample blend factor t [N]."""
    # Ensure t broadcasts correctly against [..., 4] tensors
    if t.dim() == 1:
        t = t.unsqueeze(-1)  # [N] -> [N, 1]
    dot = (a * b).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    b_adj = torch.where(dot < 0, -b, b)          # ensure shortest arc
    dot   = dot.abs()
    theta = torch.acos(dot.clamp(max=1.0 - 1e-7))
    sin_t = torch.sin(theta)
    safe  = sin_t.abs() > 1e-7
    w_a   = torch.where(safe, torch.sin((1 - t) * theta) / sin_t, (1 - t) * torch.ones_like(sin_t))
    w_b   = torch.where(safe, torch.sin(t       * theta) / sin_t, t       * torch.ones_like(sin_t))
    return w_a * a + w_b * b_adj


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------

class BonesCSVConverter:
    """Converts bones-seed G1 CSV files to NPZ via standalone FK.

    Args:
        urdf_path:  Path to the G1 URDF file.
        input_fps:  Source motion frame rate (default 120).
        output_fps: Target NPZ frame rate (default 50).
        device:     Torch device, e.g. ``'cpu'`` or ``'cuda'``.
        batch_size: Number of frames processed per FK batch.
    """

    def __init__(
        self,
        urdf_path: str,
        input_fps:  int = 120,
        output_fps: int = 50,
        device:     str = "cpu",
        batch_size: int = 512,
    ) -> None:
        try:
            import pytorch_kinematics as pk
        except ImportError as exc:
            raise ImportError(
                "pytorch-kinematics is required. Install with:\n"
                "  pip install pytorch-kinematics"
            ) from exc

        self.urdf_path  = str(urdf_path)
        self.input_fps  = input_fps
        self.output_fps = output_fps
        self.device     = device
        self.batch_size = batch_size
        self.input_dt   = 1.0 / input_fps
        self.output_dt  = 1.0 / output_fps

        # ---- Build kinematic chain ----
        with open(urdf_path, "rb") as fh:
            urdf_bytes = fh.read()

        # pytorch_kinematics .to() is in-place; SerialChain.__init__ does CPU
        # matmuls internally, so ALL chains must be constructed while the root
        # chain is still on CPU.  Move to target device only after construction.
        _chain_raw = pk.build_chain_from_urdf(urdf_bytes)

        self._chain_joint_names: list[str] = (
            _chain_raw.get_joint_parameter_names()
        )

        # Map from chain-index → MUJOCO index
        # (for G1, URDF chain order == MUJOCO_DOF_NAMES order)
        self._chain_to_mujoco: list[int] = []
        for name in self._chain_joint_names:
            if name not in MUJOCO_DOF_NAMES:
                raise ValueError(
                    f"URDF joint '{name}' not in MUJOCO_DOF_NAMES. "
                    "Ensure you are using the G1 URDF."
                )
            self._chain_to_mujoco.append(MUJOCO_DOF_NAMES.index(name))
        self._mujoco_to_chain = [0] * N_DOFS
        for chain_i, mujoco_i in enumerate(self._chain_to_mujoco):
            self._mujoco_to_chain[mujoco_i] = chain_i

        # Build SerialChains while root chain is still on CPU, then move all
        # chains to target device together.
        _serial_chains_raw = [
            pk.SerialChain(_chain_raw, bname) for bname in ISAACLAB_BODY_NAMES
        ]

        self._full_chain = _chain_raw.to(device=device, dtype=torch.float32)
        self._body_serial_chains = [
            sc.to(device=device, dtype=torch.float32) for sc in _serial_chains_raw
        ]

        self._body_mujoco_indices: list[list[int]] = []
        for sc in self._body_serial_chains:
            sc_joints = sc.get_joint_parameter_names()
            indices = [
                MUJOCO_DOF_NAMES.index(j)
                for j in sc_joints
                if j in MUJOCO_DOF_NAMES
            ]
            self._body_mujoco_indices.append(indices)

        print(
            f"[BonesCSVConverter] URDF loaded: {urdf_path}\n"
            f"  chain joints={len(self._chain_joint_names)}, "
            f"bodies={N_BODIES}, device={device}, "
            f"input_fps={input_fps}, output_fps={output_fps}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def convert_file(
        self,
        csv_path:    str | Path,
        output_path: str | Path,
    ) -> Path:
        """Convert a single bones-seed CSV to NPZ.

        Args:
            csv_path:    Path to input CSV.
            output_path: Path for the output ``.npz`` file.
                         Parent directories are created if needed.

        Returns:
            Resolved ``Path`` of the written NPZ.
        """
        csv_path    = Path(csv_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Load and parse CSV
        q_in, root_pos_in, root_quat_in = self._load_csv(csv_path)

        # 2. Resample to output_fps
        q, root_pos, root_quat = self._resample(
            q_in, root_pos_in, root_quat_in
        )
        T = q.shape[0]

        # 3. Joint velocities via gradient
        joint_vel = torch.gradient(q, spacing=(self.output_dt,), dim=0)[0]

        # 3b. Root velocities needed for full rigid-body velocity formula
        #     v_body = R @ J[:3] @ dq  +  v_root  +  ω_root × r_body
        root_lin_vel  = torch.gradient(root_pos,  spacing=(self.output_dt,), dim=0)[0]  # [T, 3]
        root_ang_vel_w = _so3_derivative_per_body(
            root_quat.unsqueeze(1), self.output_dt
        ).squeeze(1)  # [T, 3]  world-frame angular velocity of root

        # 4. Root rotation matrices [T, 3, 3]
        R_root = _quat_to_rotmat(root_quat)  # [T, 3, 3]

        # 5. Run FK in batches → body_pos_w, body_rot_base, body_lin_vel_w
        body_pos_base  = torch.zeros(T, N_BODIES, 3)
        body_rot_base  = torch.zeros(T, N_BODIES, 3, 3)
        body_lin_vel_w = torch.zeros(T, N_BODIES, 3)

        for start in range(0, T, self.batch_size):
            end   = min(start + self.batch_size, T)
            q_b   = q[start:end].to(self.device)              # [B, 29]
            dq_b  = joint_vel[start:end].to(self.device)      # [B, 29]
            R_b   = R_root[start:end].to(self.device)         # [B, 3, 3]
            rp_b  = root_pos[start:end].to(self.device)       # [B, 3]
            rlv_b = root_lin_vel[start:end].to(self.device)   # [B, 3]
            rav_b = root_ang_vel_w[start:end].to(self.device) # [B, 3]

            pb, rb, vlb = self._fk_batch(q_b, dq_b, R_b, rp_b, rlv_b, rav_b)
            body_pos_base[start:end]  = pb.cpu()
            body_rot_base[start:end]  = rb.cpu()
            body_lin_vel_w[start:end] = vlb.cpu()

        # 6. Transform body rotations to world frame → quaternions
        #    R_world[t, b] = R_root[t] @ R_base[t, b]
        R_root_exp = R_root.unsqueeze(1).expand(-1, N_BODIES, -1, -1)  # [T,30,3,3]
        R_world    = torch.bmm(
            R_root_exp.reshape(-1, 3, 3),
            body_rot_base.reshape(-1, 3, 3),
        ).reshape(T, N_BODIES, 3, 3)

        body_quat_w = _rotmat_to_quat_wxyz(R_world)  # [T, 30, 4]

        # 7. Body angular velocities via SO3 finite difference
        body_ang_vel_w = _so3_derivative_per_body(body_quat_w, self.output_dt)

        # 8. Save — joint_pos/vel in Isaac Lab DOF order (matches csv_to_npz.py /
        #    robot.data.joint_pos which is alphabetical URDF order = ISAACLAB_DOF_NAMES)
        il_idx = torch.tensor(_MUJOCO_TO_ISAACLAB, dtype=torch.long)
        q_il        = q[:, il_idx]
        jv_il       = joint_vel[:, il_idx]

        np.savez(
            output_path,
            fps=np.array([self.output_fps], dtype=np.float32),
            joint_pos=q_il.numpy().astype(np.float32),
            joint_vel=jv_il.numpy().astype(np.float32),
            body_pos_w=body_pos_base.numpy().astype(np.float32),
            body_quat_w=body_quat_w.numpy().astype(np.float32),
            body_lin_vel_w=body_lin_vel_w.numpy().astype(np.float32),
            body_ang_vel_w=body_ang_vel_w.numpy().astype(np.float32),
        )
        print(f"[BonesCSVConverter] Saved {output_path}  ({T} frames @ {self.output_fps} fps)")
        return output_path.resolve()

    def convert_batch(
        self,
        rows_df,           # pd.DataFrame with move_g1_path column
        output_dir: str | Path,
        base_path:  str | Path,
        flat: bool = False,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
    ) -> list[Path]:
        """Convert every row in *rows_df* to NPZ.

        Args:
            rows_df:     DataFrame from :meth:`BonesSeedParser.filter`.
            output_dir:  Root directory for NPZ output.
            base_path:   Dataset root (same as ``BonesSeedParser.base_path``).
            flat:        If ``True``, write ``{output_dir}/{move_name}.npz``.
                         If ``False`` (default), write ``{output_dir}/{move_name}/motion.npz``.
            progress_cb: Optional callback ``(done, total, move_name)``.

        Returns:
            List of output paths (only successfully converted files).
        """
        from tqdm import tqdm  # soft dependency

        output_dir = Path(output_dir)
        base_path  = Path(base_path)
        results: list[Path] = []
        total = len(rows_df)

        for i, (_, row) in enumerate(tqdm(rows_df.iterrows(), total=total, desc="Converting")):
            move_name = row["move_name"]
            csv_src   = (base_path / row["move_g1_path"]).resolve()

            if flat:
                out = output_dir / f"{move_name}.npz"
            else:
                out = output_dir / move_name / "motion.npz"

            if not csv_src.exists():
                print(f"[WARN] CSV not found, skipping: {csv_src}")
                if progress_cb:
                    progress_cb(i + 1, total, f"SKIP {move_name}")
                continue

            try:
                path = self.convert_file(csv_src, out)
                results.append(path)
            except Exception as exc:
                import traceback as _tb
                print(f"[ERROR] {move_name}: {exc}")
                _tb.print_exc()

            if progress_cb:
                progress_cb(i + 1, total, move_name)

        print(f"[BonesCSVConverter] Batch done: {len(results)}/{total} converted → {output_dir}")
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_csv(
        self, csv_path: Path
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Parse bones-seed CSV.

        Returns:
            joint_pos_rad: [T, 29] float32 (MUJOCO order, radians)
            root_pos_m:    [T, 3]  float32 (metres)
            root_quat_wxyz:[T, 4]  float32
        """
        # Detect header
        with open(csv_path) as fh:
            first_line = fh.readline()
        try:
            [float(v) for v in first_line.split(",")]
            skip = 0
        except ValueError:
            skip = 1

        data = torch.from_numpy(
            np.loadtxt(csv_path, delimiter=",", skiprows=skip)
        ).float()

        # col 0 = Frame (index), cols 1-3 = pos cm, cols 4-6 = Euler ZYX deg, cols 7+ = DOF deg
        root_pos_m    = data[:, 1:4] / 100.0
        root_quat     = _euler_zyx_deg_to_quat_wxyz(data[:, 4:7])
        joint_pos_rad = torch.deg2rad(data[:, 7:7 + N_DOFS])

        return joint_pos_rad, root_pos_m, root_quat

    def _resample(
        self,
        q:         torch.Tensor,   # [T_in, 29]
        root_pos:  torch.Tensor,   # [T_in, 3]
        root_quat: torch.Tensor,   # [T_in, 4]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Resample from input_fps to output_fps."""
        T_in    = q.shape[0]
        dur     = (T_in - 1) * self.input_dt
        times   = torch.arange(0.0, dur, self.output_dt)
        T_out   = times.shape[0]

        phase   = times / dur
        idx0    = (phase * (T_in - 1)).floor().long()
        idx1    = (idx0 + 1).clamp(max=T_in - 1)
        blend   = (phase * (T_in - 1) - idx0).unsqueeze(1)  # [T_out, 1]

        q_out   = q[idx0] * (1 - blend) + q[idx1] * blend

        rp_out  = root_pos[idx0] * (1 - blend) + root_pos[idx1] * blend

        # Quaternion slerp (vectorised)
        rq_out  = _slerp(root_quat[idx0], root_quat[idx1], blend[:, 0])

        return q_out, rp_out, rq_out

    def _fk_batch(
        self,
        q:            torch.Tensor,  # [B, 29] MUJOCO order (radians)
        dq:           torch.Tensor,  # [B, 29]
        R_root:       torch.Tensor,  # [B, 3, 3] world ← base rotation
        base_pos:     torch.Tensor,  # [B, 3]
        base_lin_vel: torch.Tensor,  # [B, 3] root linear velocity in world frame
        ang_vel_w:    torch.Tensor,  # [B, 3] root angular velocity in world frame
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run FK for a batch. Returns (body_pos_w, body_rot_base, body_lin_vel_w).

        Full rigid-body velocity formula (matches g1fk_torch._compute_fk_batch):
            v_body = R @ J[:3] @ dq  +  base_lin_vel  +  ω_root × r_body
        where r_body = body_pos_w − base_pos.
        """
        B   = q.shape[0]
        dev = self.device

        # Reorder from MUJOCO to chain order:
        # _chain_to_mujoco[chain_i] = mujoco_i  →  q_chain[:, chain_i] = q[:, mujoco_i]
        chain_idx = torch.tensor(self._chain_to_mujoco, device=dev, dtype=torch.long)
        q_chain   = q[:, chain_idx]   # [B, 29]

        q_dict = {
            name: q_chain[:, i]
            for i, name in enumerate(self._chain_joint_names)
        }

        with torch.no_grad():
            ret = self._full_chain.forward_kinematics(q_dict)

        body_pos_base = torch.zeros(B, N_BODIES, 3, device=dev)
        body_rot_base = torch.zeros(B, N_BODIES, 3, 3, device=dev)

        for bdy_i, bname in enumerate(ISAACLAB_BODY_NAMES):
            m = ret[bname].get_matrix().to(dev)       # [B, 4, 4]
            body_pos_base[:, bdy_i, :] = m[:, :3, 3]
            body_rot_base[:, bdy_i, :] = m[:, :3, :3]

        # Transform positions to world frame: p_w = p_base @ R.T + base_pos
        # torch.bmm([B, 30, 3], [B, 3, 3]) computes per-batch p_base[i] @ R.T
        body_pos_w = (
            torch.bmm(body_pos_base, R_root.transpose(1, 2))
            + base_pos.unsqueeze(1)
        )

        # Body linear velocities: v = R@J[:3]@dq  +  v_root  +  ω×r_body
        body_lin_vel_w = torch.zeros(B, N_BODIES, 3, device=dev)
        for bdy_i, (sc, muj_idx) in enumerate(
            zip(self._body_serial_chains, self._body_mujoco_indices)
        ):
            if not muj_idx:
                # Pelvis (root body): no joints, velocity = root translational vel
                body_lin_vel_w[:, bdy_i, :] = base_lin_vel
                continue

            # q is in MUJOCO order; muj_idx holds MUJOCO indices in sc joint order
            q_sc  = q[:, muj_idx]   # [B, n_sc]
            dq_sc = dq[:, muj_idx]
            with torch.no_grad():
                J = sc.jacobian(q_sc).to(dev)  # [B, 6, n_sc]

            # Joint-motion velocity, rotated to world frame
            vel_base  = torch.bmm(J[:, :3, :], dq_sc.unsqueeze(-1)).squeeze(-1)  # [B, 3]
            vel_world = torch.bmm(vel_base.unsqueeze(1), R_root.transpose(1, 2)).squeeze(1)

            # Angular contribution: ω_root × r_body  (r_body in world frame)
            r_body      = body_pos_w[:, bdy_i, :] - base_pos          # [B, 3]
            ang_contrib = torch.cross(ang_vel_w, r_body, dim=-1)      # [B, 3]

            body_lin_vel_w[:, bdy_i, :] = vel_world + base_lin_vel + ang_contrib

        return body_pos_w, body_rot_base, body_lin_vel_w
