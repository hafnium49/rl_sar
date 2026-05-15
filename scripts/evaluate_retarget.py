#!/usr/bin/env python3
"""Evaluate retargeting quality for the Motion Star → Unitree G1 pipeline.

Outputs a JSON report scoring the retarget against thresholds in the second-opinion review.
Exit 0 if all hard-fail thresholds pass; exit 1 otherwise — the autonomous loop driver gates on
this exit code.

Five priority metrics (extensible by editing the metric-computation block below):
  1. Weighted keypoint MPJPE (with pelvis alignment) — normalized by G1 height
  2. Foot contact slip during source-contact frames
  3. Joint-limit violation rate
  4. Joint jump rate (per-frame Δq > 0.25 rad)
  5. Self-collision + ground-penetration frame rate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

from general_motion_retargeting.params import ROBOT_XML_DICT

# Hard-fail thresholds = autonomous gates. Soft = observability only (logged, not gating).
# Thresholds calibrated for 30-fps Ibaraki Motion Star data + G1's foot-mesh geometry.
DEFAULT_THRESHOLDS = {
    "weighted_keypoint_error_pct_height":  {"max": 6.0,  "severity": "hard"},
    "pelvis_error_cm":                     {"max": 4.0,  "severity": "hard"},
    "foot_error_during_contact_cm":        {"max": 5.0,  "severity": "hard"},
    "wrist_error_cm":                      {"max": 12.0, "severity": "soft"},
    "joint_limit_violation_frame_pct":     {"max": 0.5,  "severity": "hard"},
    "joint_jump_frame_pct":                {"max": 3.0,  "severity": "hard"},
    "self_collision_frame_pct":            {"max": 3.0,  "severity": "hard"},
    "ground_penetration_frame_pct":        {"max": 1.0,  "severity": "hard"},
    "foot_slip_during_contact_cm_per_s":   {"max": 5.0,  "severity": "hard"},
}

# Priority-based body weights for MPJPE; physics-critical >> style >> expressive.
BODY_WEIGHTS = {
    "pelvis": 5.0,
    "left_toe_link": 8.0,           "right_toe_link": 8.0,
    "torso_link": 4.0,
    "left_knee_link": 2.0,          "right_knee_link": 2.0,
    "left_hip_roll_link": 2.0,      "right_hip_roll_link": 2.0,
    "left_elbow_link": 2.0,         "right_elbow_link": 2.0,
    "left_wrist_yaw_link": 3.0,     "right_wrist_yaw_link": 3.0,
    "left_rubber_hand": 3.0,        "right_rubber_hand": 3.0,
    "left_shoulder_yaw_link": 2.0,  "right_shoulder_yaw_link": 2.0,
}

G1_HEIGHT_M = 1.5
FOOT_CONTACT_Z = 0.15          # m — chosen to capture rest-pose foot height (~0.09 m) + margin
FOOT_CONTACT_VEL = 0.10        # m/s — generous; mocap sensors have small residual noise even at rest
JOINT_JUMP_RAD = 0.50          # per frame at 30 fps; 50 fps reference of 0.25 → 0.42 here, rounded up
NEAR_LIMIT_RAD = np.deg2rad(5.0)
PENETRATION_M = 0.005          # ignore tiny contact distances below this (numerical noise)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-npz", type=Path, required=True)
    ap.add_argument("--g1-qpos",    type=Path, required=True)
    ap.add_argument("--mapping",    type=Path, required=True)
    ap.add_argument("--output",     type=Path, required=True)
    ap.add_argument("--thresholds", type=Path, default=None,
                    help="optional JSON file overriding DEFAULT_THRESHOLDS")
    args = ap.parse_args()

    thresholds = (json.loads(args.thresholds.read_text()) if args.thresholds else DEFAULT_THRESHOLDS)

    # ---- Load inputs ----
    src = np.load(args.source_npz)
    src_pos_all = src["pos"]       # (T_src, 15, 3) Z-up m
    src_names = list(src["sensor_names"])
    g1 = np.load(args.g1_qpos)
    g1_qpos = g1["qpos"]           # (T_g1, 36)
    n_g1 = g1_qpos.shape[0]

    # Retargeter skips source frame 0 (empty fcurve eval); align src[1:] ↔ g1[0:]
    src_pos = src_pos_all[1:1 + n_g1]
    n_frames = src_pos.shape[0]
    assert n_frames == n_g1, f"frame alignment broken: src={n_frames} g1={n_g1}"

    cfg = json.loads(args.mapping.read_text())
    body_to_sensor: dict[str, str] = {}
    for body_name, entry in cfg["ik_match_table2"].items():
        if body_name.startswith("_"):
            continue
        body_to_sensor[body_name] = entry[0]

    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT["unitree_g1"]))
    data = mujoco.MjData(model)

    body_ids: dict[str, int] = {}
    for bn in body_to_sensor.keys():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bn)
        if bid >= 0:
            body_ids[bn] = bid
    body_names_list = list(body_ids.keys())
    body_id_arr = np.array([body_ids[bn] for bn in body_names_list], dtype=int)

    # Joint-limit lookup once (skip floating-base root)
    joint_limits: list[tuple[int, float, float]] = []
    for j in range(model.njnt):
        if not model.jnt_limited[j]:
            continue
        qa = model.jnt_qposadr[j]
        if qa < 7:
            continue
        joint_limits.append((qa, float(model.jnt_range[j, 0]), float(model.jnt_range[j, 1])))

    print(f"[eval] n_frames={n_frames}  bodies={len(body_names_list)}  joint_limits={len(joint_limits)}")

    # ---- Single pass: FK + contacts + per-frame body positions ----
    g1_body_pos = np.zeros((n_frames, len(body_names_list), 3), dtype=np.float64)
    self_coll_frames = np.zeros(n_frames, dtype=bool)
    ground_pen_frames = np.zeros(n_frames, dtype=bool)
    for f in range(n_frames):
        data.qpos[:] = g1_qpos[f]
        mujoco.mj_forward(model, data)
        g1_body_pos[f] = data.xpos[body_id_arr]
        for k in range(data.ncon):
            c = data.contact[k]
            b1 = int(model.geom_bodyid[c.geom1])
            b2 = int(model.geom_bodyid[c.geom2])
            if c.dist >= -PENETRATION_M:
                continue
            if b1 == 0 or b2 == 0:
                ground_pen_frames[f] = True
            else:
                self_coll_frames[f] = True

    # ---- Metric 1: weighted keypoint MPJPE (pelvis-aligned) ----
    src_body_pos = np.zeros_like(g1_body_pos)
    for i, bn in enumerate(body_names_list):
        sname = body_to_sensor[bn]
        if sname in src_names:
            src_body_pos[:, i] = src_pos[:, src_names.index(sname)]

    pelvis_i = body_names_list.index("pelvis") if "pelvis" in body_names_list else 0
    pelvis_offset = src_body_pos[0, pelvis_i] - g1_body_pos[0, pelvis_i]
    src_body_pos_aligned = src_body_pos - pelvis_offset[None, None, :]

    err_per_frame = np.linalg.norm(g1_body_pos - src_body_pos_aligned, axis=2)  # (T, B)
    weights = np.array([BODY_WEIGHTS.get(bn, 1.0) for bn in body_names_list])
    weighted_per_frame = (err_per_frame * weights[None, :]).sum(axis=1) / weights.sum()
    weighted_keypoint_error_pct_height = float((weighted_per_frame.mean() / G1_HEIGHT_M) * 100.0)

    per_body_error_cm = {bn: float(err_per_frame[:, i].mean() * 100.0)
                         for i, bn in enumerate(body_names_list)}
    pelvis_error_cm = per_body_error_cm.get("pelvis", float("nan"))
    wrist_error_cm = max(
        per_body_error_cm.get("left_rubber_hand", per_body_error_cm.get("left_wrist_yaw_link", 0.0)),
        per_body_error_cm.get("right_rubber_hand", per_body_error_cm.get("right_wrist_yaw_link", 0.0)),
    )

    # ---- Metric 2: foot contact slip ----
    def velocity_xy(pos_t: np.ndarray) -> np.ndarray:
        v = np.zeros(pos_t.shape[0])
        v[1:] = np.linalg.norm(pos_t[1:, :2] - pos_t[:-1, :2], axis=1) * 30.0
        return v

    src_lf = src_pos[:, src_names.index("Sensor1")]
    src_rf = src_pos[:, src_names.index("Sensor2")]
    left_contact = (src_lf[:, 2] < FOOT_CONTACT_Z) & (velocity_xy(src_lf) < FOOT_CONTACT_VEL)
    right_contact = (src_rf[:, 2] < FOOT_CONTACT_Z) & (velocity_xy(src_rf) < FOOT_CONTACT_VEL)

    def foot_slip_cm_per_s(g1_idx_name: str, contact_mask: np.ndarray) -> float:
        if g1_idx_name not in body_names_list or not contact_mask.any():
            return 0.0
        gi = body_names_list.index(g1_idx_name)
        vel = velocity_xy(g1_body_pos[:, gi])
        return float(vel[contact_mask].mean() * 100.0)

    left_slip = foot_slip_cm_per_s("left_toe_link", left_contact)
    right_slip = foot_slip_cm_per_s("right_toe_link", right_contact)
    foot_slip_during_contact_cm_per_s = max(left_slip, right_slip)

    def foot_err_cm(g1_idx_name: str, contact_mask: np.ndarray) -> float:
        if g1_idx_name not in body_names_list or not contact_mask.any():
            return 0.0
        gi = body_names_list.index(g1_idx_name)
        e = np.linalg.norm(g1_body_pos[contact_mask, gi, :2] -
                           src_body_pos_aligned[contact_mask, gi, :2], axis=1)
        return float(e.mean() * 100.0)

    foot_error_during_contact_cm = max(
        foot_err_cm("left_toe_link", left_contact),
        foot_err_cm("right_toe_link", right_contact),
    )

    # ---- Metric 3: joint-limit violation rate ----
    violation_frames = np.zeros(n_frames, dtype=bool)
    near_limit_frames = np.zeros(n_frames, dtype=bool)
    max_violation_rad = 0.0
    for qa, lo, hi in joint_limits:
        q = g1_qpos[:, qa]
        violation_frames |= (q < lo) | (q > hi)
        near_limit_frames |= ((q > lo) & (q < lo + NEAR_LIMIT_RAD)) | ((q > hi - NEAR_LIMIT_RAD) & (q < hi))
        v = np.maximum(lo - q, q - hi)
        if v.max() > 0:
            max_violation_rad = max(max_violation_rad, float(v.max()))
    joint_limit_violation_frame_pct = float(100 * violation_frames.mean())
    near_limit_frame_pct = float(100 * near_limit_frames.mean())

    # ---- Metric 4: joint jumps ----
    dq = np.abs(g1_qpos[1:, 7:] - g1_qpos[:-1, 7:])  # (T-1, 29)
    max_per_frame = dq.max(axis=1)
    joint_jump_frame_pct = float(100 * (max_per_frame > JOINT_JUMP_RAD).mean())
    max_joint_delta_rad = float(max_per_frame.max())

    # ---- Metric 5: collisions ----
    self_collision_frame_pct = float(100 * self_coll_frames.mean())
    ground_penetration_frame_pct = float(100 * ground_pen_frames.mean())

    # ---- Assemble report ----
    report = {
        "n_frames": int(n_frames),
        "weighted_keypoint_error_pct_height": weighted_keypoint_error_pct_height,
        "pelvis_error_cm": pelvis_error_cm,
        "foot_error_during_contact_cm": foot_error_during_contact_cm,
        "wrist_error_cm": wrist_error_cm,
        "joint_limit_violation_frame_pct": joint_limit_violation_frame_pct,
        "near_limit_frame_pct": near_limit_frame_pct,
        "max_violation_rad": max_violation_rad,
        "joint_jump_frame_pct": joint_jump_frame_pct,
        "max_joint_delta_rad_per_frame": max_joint_delta_rad,
        "self_collision_frame_pct": self_collision_frame_pct,
        "ground_penetration_frame_pct": ground_penetration_frame_pct,
        "foot_slip_during_contact_cm_per_s": foot_slip_during_contact_cm_per_s,
        "left_foot_contact_frames": int(left_contact.sum()),
        "right_foot_contact_frames": int(right_contact.sum()),
        "per_body_error_cm": per_body_error_cm,
    }

    pass_flags: dict[str, bool] = {}
    all_hard_pass = True
    for k, th in thresholds.items():
        if k not in report:
            pass_flags[k] = False
            continue
        passed = report[k] <= th["max"]
        pass_flags[k] = passed
        if th["severity"] == "hard" and not passed:
            all_hard_pass = False
    report["pass_flags"] = pass_flags
    report["all_hard_pass"] = bool(all_hard_pass)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))

    # ---- Console summary ----
    print(f"=== {args.output.name} ===")
    for k in ("weighted_keypoint_error_pct_height", "pelvis_error_cm",
              "foot_error_during_contact_cm", "wrist_error_cm",
              "joint_limit_violation_frame_pct", "joint_jump_frame_pct",
              "self_collision_frame_pct", "ground_penetration_frame_pct",
              "foot_slip_during_contact_cm_per_s"):
        v = report[k]
        th = thresholds.get(k, {})
        flag = "PASS" if pass_flags.get(k) else ("SOFT" if th.get("severity") == "soft" else "FAIL")
        thresh_str = f"<= {th.get('max', '?')}"
        print(f"  {k:42s} = {v:7.3f}  {thresh_str:12s}  [{flag}]")
    print(f"--- {'ALL HARD PASS' if all_hard_pass else 'HARD THRESHOLD FAILURE'} ---")
    return 0 if all_hard_pass else 1


if __name__ == "__main__":
    sys.exit(main())
