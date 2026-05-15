#!/usr/bin/env python3
"""Mirror-equivariance test for the Motion Star → Unitree G1 retargeter.

Tests whether the retargeter treats left and right body chains consistently
("morphological equivariance"). The test:
    F(M_source(x)) ≈ M_robot(F(x))
where F is the retargeter, M_source mirrors the source NPZ across the sagittal plane,
and M_robot mirrors the resulting G1 body positions across the same plane.

Acceptance: max per-body position difference < 5 cm = retargeter is unbiased.

This is the lightweight diagnostic from the third-opinion review's framework. It catches
left-right mapping bias (one-side-worse output despite mathematically symmetric IK weights).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

from general_motion_retargeting.params import ROBOT_XML_DICT

# MuJoCo G1 convention: X=forward, Y=left, Z=up. Sagittal plane mirror = flip Y.
# Source NPZ is in the same (right-handed Z-up) frame after Iter 2's reflection fix,
# so we flip Y on the source positions as well.

SENSOR_PAIRS = [
    (1, 2),    # left_foot / right_foot
    (3, 4),    # left_shin / right_shin
    (5, 6),    # left_thigh / right_thigh
    (8, 9),    # right_hand / left_hand  (Sensor8 = right, Sensor9 = left)
    (10, 11),  # right_forearm / left_forearm
    (12, 13),  # right_upper_arm / left_upper_arm
]

BODY_PAIRS = [
    ("left_toe_link",          "right_toe_link"),
    ("left_hip_roll_link",     "right_hip_roll_link"),
    ("left_knee_link",         "right_knee_link"),
    ("left_shoulder_yaw_link", "right_shoulder_yaw_link"),
    ("left_elbow_link",        "right_elbow_link"),
    ("left_rubber_hand",       "right_rubber_hand"),
]
SELF_PAIRED_BODIES = ["pelvis", "torso_link"]


def mirror_source_npz(in_path: Path, out_path: Path) -> None:
    """Mirror a source NPZ by flipping Y axis + swapping left/right sensor indices."""
    src = np.load(in_path)
    pos = src["pos"].copy()    # (T, 15, 3)
    quat = src["quat"].copy()  # (T, 15, 4)

    pos[..., 1] *= -1  # Y flip mirrors across sagittal plane

    # Sensor indices are 1-based names ("Sensor1", "Sensor2", ...); array index = N-1
    for a, b in SENSOR_PAIRS:
        idx_a, idx_b = a - 1, b - 1
        pos[:, [idx_a, idx_b]] = pos[:, [idx_b, idx_a]]
        quat[:, [idx_a, idx_b]] = quat[:, [idx_b, idx_a]]
    # Quaternions are garbage from Iter 2's reflection anyway — IK ignores them — so we
    # don't need a principled rotation mirror here. Swapping L/R quaternions is sufficient
    # to keep the data structure consistent.

    np.savez_compressed(
        out_path,
        pos=pos.astype(np.float32),
        quat=quat.astype(np.float32),
        frame_time=src["frame_time"],
        n_frames=src["n_frames"],
        sensor_names=src["sensor_names"],
    )


def run_retarget(src_npz: Path, out_qpos: Path) -> None:
    """Invoke motionstar_retarget.py via subprocess."""
    cmd = [
        "/home/h_fujiwara/miniconda3/bin/conda", "run", "-n", "gmr", "python",
        str(Path(__file__).parent / "motionstar_retarget.py"),
        "--npz", str(src_npz), "--out", str(out_qpos),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout); print(r.stderr, file=sys.stderr)
        raise RuntimeError(f"retarget failed (exit {r.returncode})")


def body_xpos_per_frame(qpos: np.ndarray, body_ids: list[int]) -> np.ndarray:
    """Run mj_forward per frame and return (T, len(body_ids), 3) world positions."""
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT["unitree_g1"]))
    data = mujoco.MjData(model)
    T = qpos.shape[0]
    out = np.zeros((T, len(body_ids), 3))
    for f in range(T):
        data.qpos[:] = qpos[f]
        mujoco.mj_forward(model, data)
        for i, bid in enumerate(body_ids):
            out[f, i] = data.xpos[bid]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--take", type=int, default=12)
    ap.add_argument("--output", type=Path,
                    default=Path("/home/h_fujiwara/datasets/radio_taiso/reports/manual/equivariance.json"))
    args = ap.parse_args()

    src_orig = Path(f"/home/h_fujiwara/datasets/radio_taiso/intermediate/motionstar_npz/take{args.take}.npz")
    src_mir = Path(f"/home/h_fujiwara/datasets/radio_taiso/intermediate/motionstar_npz/take{args.take}_mirrored.npz")
    qpos_orig_path = Path(f"/home/h_fujiwara/datasets/radio_taiso/gmr/take{args.take}_g1.npz")
    qpos_mir_path = Path(f"/home/h_fujiwara/datasets/radio_taiso/gmr/take{args.take}_g1_mirrored.npz")

    print(f"[equivariance] mirroring source NPZ: {src_orig} → {src_mir}")
    mirror_source_npz(src_orig, src_mir)

    print(f"[equivariance] running retarget on mirrored source...")
    run_retarget(src_mir, qpos_mir_path)

    qpos_orig = np.load(qpos_orig_path)["qpos"]
    qpos_mir = np.load(qpos_mir_path)["qpos"]
    print(f"[equivariance] qpos shapes: orig={qpos_orig.shape}, mirrored={qpos_mir.shape}")
    T = min(qpos_orig.shape[0], qpos_mir.shape[0])
    qpos_orig = qpos_orig[:T]
    qpos_mir = qpos_mir[:T]

    # Look up body IDs in the order: [pelvis, torso_link, L_toe, R_toe, L_hip, R_hip, ...]
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT["unitree_g1"]))
    body_names_ordered: list[str] = []
    body_names_ordered.extend(SELF_PAIRED_BODIES)
    for left, right in BODY_PAIRS:
        body_names_ordered.extend([left, right])
    body_ids = []
    for bn in body_names_ordered:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bn)
        if bid < 0:
            print(f"WARNING: body {bn} not found in G1 model")
        body_ids.append(bid)
    valid = [(i, bn) for i, (bid, bn) in enumerate(zip(body_ids, body_names_ordered)) if bid >= 0]
    body_ids = [body_ids[i] for i, _ in valid]
    body_names_ordered = [bn for _, bn in valid]

    print(f"[equivariance] computing body xpos via mj_forward (orig + mirrored)...")
    xpos_orig = body_xpos_per_frame(qpos_orig, body_ids)  # (T, B, 3)
    xpos_mir = body_xpos_per_frame(qpos_mir, body_ids)

    # Construct the "mirrored expected" xpos from xpos_orig:
    #   - For self-paired bodies (pelvis, torso): flip Y of their own position
    #   - For L/R pairs: mirror_expected[L_idx] = flip_Y(xpos_orig[R_idx]) and vice versa
    # The mirrored retarget should match this.
    n_self = len(SELF_PAIRED_BODIES)
    xpos_mirror_expected = xpos_orig.copy()
    # Flip Y on self-paired bodies
    for i in range(n_self):
        xpos_mirror_expected[:, i, 1] *= -1
    # For L/R pairs: positions at i (left) come from position at i+1 (right) with Y flipped, vice versa
    for j, (left_bn, right_bn) in enumerate(BODY_PAIRS):
        # Find indices of left and right bodies in body_names_ordered
        try:
            l_idx = body_names_ordered.index(left_bn)
            r_idx = body_names_ordered.index(right_bn)
        except ValueError:
            continue
        xpos_mirror_expected[:, l_idx] = xpos_orig[:, r_idx] * np.array([1, -1, 1])
        xpos_mirror_expected[:, r_idx] = xpos_orig[:, l_idx] * np.array([1, -1, 1])

    # Compute per-body diff: F(M(x)) vs M(F(x))
    diff = np.linalg.norm(xpos_mir - xpos_mirror_expected, axis=2)  # (T, B)
    per_body_mean_cm = (diff.mean(axis=0) * 100).tolist()
    per_body_max_cm = (diff.max(axis=0) * 100).tolist()
    max_body_idx = int(np.argmax(diff.max(axis=0)))
    overall_max_cm = float(diff.max() * 100)

    report = {
        "take": args.take,
        "n_frames": int(T),
        "overall_max_diff_cm": overall_max_cm,
        "worst_body": body_names_ordered[max_body_idx],
        "per_body_mean_cm": dict(zip(body_names_ordered, per_body_mean_cm)),
        "per_body_max_cm": dict(zip(body_names_ordered, per_body_max_cm)),
        "acceptance_threshold_cm": 5.0,
        "equivariant": overall_max_cm < 5.0,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))

    print(f"\n=== Mirror-equivariance test ===")
    print(f"Frames analyzed: {T}")
    print(f"Per-body mean diff (cm):")
    for bn in body_names_ordered:
        m = report["per_body_mean_cm"][bn]
        mx = report["per_body_max_cm"][bn]
        print(f"  {bn:30s}: mean={m:6.2f}  max={mx:6.2f}")
    print(f"\nOverall max diff: {overall_max_cm:.2f} cm  (worst body: {report['worst_body']})")
    print(f"Threshold:        5.00 cm")
    print(f"Verdict: {'PASS (retargeter is equivariant)' if report['equivariant'] else 'FAIL (mapping bias detected)'}")
    return 0 if report["equivariant"] else 1


if __name__ == "__main__":
    sys.exit(main())
