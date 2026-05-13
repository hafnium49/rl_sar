#!/usr/bin/env python3
"""Record dance_102 with a camera that orbits the robot.

Same offscreen pipeline as record_dance_offscreen.py (mujoco.Renderer + cv2),
but each frame the camera's azimuth sweeps from --az-start to --az-end and the
lookat tracks the robot's root position so it stays centered.

Usage:
    /tmp/rl_sar_uv_env/bin/python scripts/record_dance_orbit.py
    /tmp/rl_sar_uv_env/bin/python scripts/record_dance_orbit.py \
        --revolutions 2 --distance 3.0 --elevation -10
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path("/home/h_fujiwara/projects/rl_sar")
SCENE = REPO / "src/rl_sar_zoo/g1_description/mjcf/scene_29dof.xml"
MOTION = REPO / "policy/g1/whole_body_tracking/dance_102/G1_Take_102.bvh_60hz.csv"
OUTPUT = REPO / "dance_102_orbit.mp4"


def load_motion(path: Path) -> np.ndarray:
    """Same CSV format as motion_loader.cpp: 36 cols, xyzw → wxyz permute."""
    rows = []
    with open(path) as f:
        for raw in csv.reader(f):
            try:
                v = [float(x) for x in raw]
            except ValueError:
                continue
            if len(v) < 36:
                continue
            v[3], v[4], v[5], v[6] = v[6], v[3], v[4], v[5]
            rows.append(v)
    return np.asarray(rows, dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--output", type=Path, default=OUTPUT)
    ap.add_argument("--scene", type=Path, default=SCENE)
    ap.add_argument("--motion", type=Path, default=MOTION)
    ap.add_argument("--distance", type=float, default=3.2,
                    help="Camera distance from the robot (meters)")
    ap.add_argument("--elevation", type=float, default=-12.0,
                    help="Camera elevation in degrees (negative looks up)")
    ap.add_argument("--az-start", type=float, default=-130.0,
                    help="Starting azimuth in degrees (matches scene default)")
    ap.add_argument("--revolutions", type=float, default=1.0,
                    help="How many full 360° loops over the whole motion")
    ap.add_argument("--lookat-z", type=float, default=0.6,
                    help="Z offset above the root for the lookat point (meters)")
    args = ap.parse_args()

    if not args.scene.exists():
        print(f"ERROR: {args.scene} not found", file=sys.stderr)
        return 2
    if not args.motion.exists():
        print(f"ERROR: {args.motion} not found", file=sys.stderr)
        return 2

    print(f"[orbit] MUJOCO_GL = {os.environ.get('MUJOCO_GL')}")
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)

    # Expand offscreen framebuffer to fit requested resolution.
    model.vis.global_.offwidth = max(args.width, int(model.vis.global_.offwidth))
    model.vis.global_.offheight = max(args.height, int(model.vis.global_.offheight))

    motion = load_motion(args.motion)
    print(f"[orbit] motion frames={len(motion)}  duration={len(motion)/args.fps:.2f}s")
    print(f"[orbit] orbit: distance={args.distance}m  elevation={args.elevation}°  "
          f"revolutions={args.revolutions}")

    renderer = mujoco.Renderer(model, args.height, args.width)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.distance = args.distance
    cam.elevation = args.elevation

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, args.fps,
                             (args.width, args.height))
    if not writer.isOpened():
        print(f"ERROR: failed to open VideoWriter for {args.output}", file=sys.stderr)
        return 3

    N = len(motion)
    sweep_deg = 360.0 * args.revolutions
    t0 = time.monotonic()
    for i, qpos in enumerate(motion):
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)

        # Look at the robot's pelvis (root + small Z lift so the torso, not the
        # feet, is at the optical center).
        cam.lookat[0] = data.qpos[0]
        cam.lookat[1] = data.qpos[1]
        cam.lookat[2] = data.qpos[2] + args.lookat_z

        # Sweep azimuth linearly across the whole motion.
        progress = i / max(N - 1, 1)
        cam.azimuth = args.az_start + progress * sweep_deg

        renderer.update_scene(data, camera=cam)
        frame_rgb = renderer.render()
        writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

        if i % 100 == 0:
            print(f"[orbit] frame {i}/{N}  az={cam.azimuth:7.1f}°  "
                  f"({100.0 * i / N:5.1f}%)")

    writer.release()
    renderer.close()
    dt = time.monotonic() - t0
    print(f"[orbit] done in {dt:.1f}s — {N} frames, {N/dt:.1f} fps  -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
