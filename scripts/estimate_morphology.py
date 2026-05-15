#!/usr/bin/env python3
"""Measure body-segment lengths for the source human (from NPZ frame 0) and the Unitree G1
(via FK on qpos0), then compute the per-segment scale ratios used by GMR's human_scale_table.

The scale ratio for a sensor → body mapping should be approximately:
    scale[sensor] = g1_segment_length / human_segment_length

For mid-segment sensors (e.g. Sensor13 = mid-upper-arm), this isn't a strict anatomical
ratio — it's the ratio of analogous segment proxies measured the same way on both skeletons.
That's what GMR's human_scale_table multiplies the local positions by (per body part, in the
root-relative frame), so a consistent proxy on both sides is what we need.

Output: ~/datasets/radio_taiso/reports/morphology.json with measurements + ratios.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

from general_motion_retargeting.params import ROBOT_XML_DICT


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-npz", type=Path,
                    default=Path("/home/h_fujiwara/datasets/radio_taiso/intermediate/motionstar_npz/take11.npz"))
    ap.add_argument("--output", type=Path,
                    default=Path("/home/h_fujiwara/datasets/radio_taiso/reports/morphology.json"))
    args = ap.parse_args()

    # ---- Source measurements from NPZ frame 0 ----
    src = np.load(args.source_npz)
    src_pos = src["pos"][0]               # (15, 3) Z-up meters
    src_names = list(src["sensor_names"])

    def s(name: str) -> np.ndarray:
        return src_pos[src_names.index(name)]

    # Source segment lengths (Euclidean distances between sensor pairs)
    # Sensor convention from earlier inspection:
    #   1/2 = L/R foot, 3/4 = L/R shin, 5/6 = L/R thigh, 7 = pelvis,
    #   8/9 = R/L hand, 10/11 = R/L forearm, 12/13 = R/L upper-arm,
    #   14 = chest, 15 = head
    src_leg_left   = np.linalg.norm(s("Sensor7") - s("Sensor1"))   # pelvis → L foot
    src_leg_right  = np.linalg.norm(s("Sensor7") - s("Sensor2"))   # pelvis → R foot
    src_arm_left   = np.linalg.norm(s("Sensor13") - s("Sensor9"))  # L upper-arm → L hand
    src_arm_right  = np.linalg.norm(s("Sensor12") - s("Sensor8"))  # R upper-arm → R hand
    src_torso      = np.linalg.norm(s("Sensor14") - s("Sensor7"))  # pelvis → chest
    src_body_height = s("Sensor15")[2] - s("Sensor1")[2]           # head Z − foot Z

    src_leg = 0.5 * (src_leg_left + src_leg_right)
    src_arm = 0.5 * (src_arm_left + src_arm_right)

    # ---- G1 measurements via FK on qpos0 ----
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT["unitree_g1"]))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    def b(name: str) -> np.ndarray:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise KeyError(f"G1 body not found: {name}")
        return data.xpos[bid].copy()

    g1_leg_left   = np.linalg.norm(b("pelvis") - b("left_toe_link"))
    g1_leg_right  = np.linalg.norm(b("pelvis") - b("right_toe_link"))
    g1_arm_left   = np.linalg.norm(b("left_shoulder_yaw_link") - b("left_rubber_hand"))
    g1_arm_right  = np.linalg.norm(b("right_shoulder_yaw_link") - b("right_rubber_hand"))
    g1_torso      = np.linalg.norm(b("torso_link") - b("pelvis"))
    # G1 standing height: rough proxy = pelvis height + torso + head allowance
    # For a more honest measurement, use the highest body z to lowest body z at qpos0:
    all_xpos = np.array([data.xpos[i] for i in range(model.nbody)])
    g1_body_height = float(all_xpos[:, 2].max() - all_xpos[:, 2].min())

    g1_leg = 0.5 * (g1_leg_left + g1_leg_right)
    g1_arm = 0.5 * (g1_arm_left + g1_arm_right)

    # ---- Ratios ----
    leg_ratio = g1_leg / src_leg
    arm_ratio = g1_arm / src_arm
    torso_ratio = g1_torso / src_torso
    body_height_ratio = g1_body_height / src_body_height

    # Sanity: human heights typically 1.4–1.9 m; G1 ~1.3 m.
    if not (1.0 < src_body_height < 2.2):
        print(f"WARNING: source body height {src_body_height:.2f} m looks unusual", file=sys.stderr)

    # ---- Report ----
    report = {
        "source": {
            "body_height_m": float(src_body_height),
            "leg_length_m_left":  float(src_leg_left),
            "leg_length_m_right": float(src_leg_right),
            "leg_length_m_avg":   float(src_leg),
            "arm_length_m_left":  float(src_arm_left),
            "arm_length_m_right": float(src_arm_right),
            "arm_length_m_avg":   float(src_arm),
            "torso_length_m":     float(src_torso),
        },
        "g1": {
            "body_height_m":      float(g1_body_height),
            "leg_length_m_left":  float(g1_leg_left),
            "leg_length_m_right": float(g1_leg_right),
            "leg_length_m_avg":   float(g1_leg),
            "arm_length_m_left":  float(g1_arm_left),
            "arm_length_m_right": float(g1_arm_right),
            "arm_length_m_avg":   float(g1_arm),
            "torso_length_m":     float(g1_torso),
        },
        "ratios_g1_over_human": {
            "leg":         float(leg_ratio),
            "arm":         float(arm_ratio),
            "torso":       float(torso_ratio),
            "body_height": float(body_height_ratio),
        },
        "sensor_to_scale_recommended": {
            # leg-related sensors get leg_ratio
            "Sensor1": float(leg_ratio),
            "Sensor2": float(leg_ratio),
            "Sensor3": float(leg_ratio),
            "Sensor4": float(leg_ratio),
            "Sensor5": float(leg_ratio),
            "Sensor6": float(leg_ratio),
            "Sensor7": float(leg_ratio),   # pelvis (root) tied to legs for ground contact
            # torso
            "Sensor14": float(torso_ratio),
            # arms
            "Sensor8":  float(arm_ratio),
            "Sensor9":  float(arm_ratio),
            "Sensor10": float(arm_ratio),
            "Sensor11": float(arm_ratio),
            "Sensor12": float(arm_ratio),
            "Sensor13": float(arm_ratio),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))

    print(f"=== morphology.json ===")
    print(f"  source body height:  {src_body_height:6.3f} m   (avg leg {src_leg:.3f}, avg arm {src_arm:.3f}, torso {src_torso:.3f})")
    print(f"  G1     body height:  {g1_body_height:6.3f} m   (avg leg {g1_leg:.3f}, avg arm {g1_arm:.3f}, torso {g1_torso:.3f})")
    print(f"  ratios (G1/human):   leg={leg_ratio:.3f}  arm={arm_ratio:.3f}  torso={torso_ratio:.3f}  body_height={body_height_ratio:.3f}")
    print(f"  → wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
