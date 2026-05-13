#!/usr/bin/env python3
"""Record the dance_102 reference motion in MuJoCo using offscreen rendering.

Inspired by the recording pattern at
/home/h_fujiwara/projects/so101-nmpc-control/docs/reference_record_script.py
(mujoco.Renderer + cv2.VideoWriter, no GLFW/X11 viewer).

Unlike `run_dance_headless.py` which captures the GLFW window via x11grab, this
script does NOT run the C++ rl_sim_mujoco binary. It does a pure kinematic
playback of the BVH-derived reference motion that the policy was trained to
track — so the robot dances cleanly without physics-induced falls, and the
output is GPU-rendered by NVIDIA EGL at native resolution.

Usage:
    /tmp/rl_sar_uv_env/bin/python scripts/record_dance_offscreen.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

# EGL backend uses the NVIDIA GB10 GPU directly — no X11, no llvmpipe.
os.environ.setdefault("MUJOCO_GL", "egl")

import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path("/home/h_fujiwara/projects/rl_sar")
SCENE = REPO / "src/rl_sar_zoo/g1_description/mjcf/scene_29dof.xml"
MOTION = REPO / "policy/g1/whole_body_tracking/dance_102/G1_Take_102.bvh_60hz.csv"
OUTPUT = REPO / "dance_102_offscreen.mp4"


def load_motion(path: Path) -> np.ndarray:
    """Load the BVH-derived CSV. Mirrors motion_loader.cpp:LoadFromCSV.

    Each row: [root_x, root_y, root_z, qx, qy, qz, qw, j0..j28]   (36 cols).
    Returns array of shape (N, 36) with quaternion permuted to MuJoCo's wxyz.
    """
    rows = []
    with open(path) as f:
        for raw in csv.reader(f):
            try:
                v = [float(x) for x in raw]
            except ValueError:
                continue
            if len(v) < 36:
                continue
            # CSV is xyzw; MuJoCo qpos wants wxyz for free-joint orientation.
            v[3], v[4], v[5], v[6] = v[6], v[3], v[4], v[5]
            rows.append(v)
    return np.asarray(rows, dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=60,
                    help="Output framerate (motion is captured at 60 Hz)")
    ap.add_argument("--output", type=Path, default=OUTPUT)
    ap.add_argument("--scene", type=Path, default=SCENE)
    ap.add_argument("--motion", type=Path, default=MOTION)
    ap.add_argument("--camera", default=None,
                    help="Named camera in the MJCF; default uses the free camera "
                         "with the scene XML's <global azimuth/elevation>.")
    args = ap.parse_args()

    if not args.scene.exists():
        print(f"ERROR: scene XML not found: {args.scene}", file=sys.stderr)
        return 2
    if not args.motion.exists():
        print(f"ERROR: motion CSV not found: {args.motion}", file=sys.stderr)
        return 2

    print(f"[record] MUJOCO_GL = {os.environ.get('MUJOCO_GL')}")
    print(f"[record] loading scene: {args.scene}")
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)

    # The scene XML's <visual><global> sets offwidth=640/offheight=480 by default.
    # mujoco.Renderer refuses to allocate larger than that, so expand here.
    model.vis.global_.offwidth = max(args.width, int(model.vis.global_.offwidth))
    model.vis.global_.offheight = max(args.height, int(model.vis.global_.offheight))

    print(f"[record] model nq={model.nq}  nv={model.nv}  njnt={model.njnt}  "
          f"offFB={model.vis.global_.offwidth}x{model.vis.global_.offheight}")

    print(f"[record] loading motion: {args.motion}")
    motion = load_motion(args.motion)
    print(f"[record] motion frames={len(motion)}  duration={len(motion)/args.fps:.2f}s")

    # 7 root + 29 joints = 36 qpos slots, matching model.nq for a free-joint
    # humanoid with 29 hinges.
    if motion.shape[1] != model.nq:
        print(f"[record] WARN motion cols ({motion.shape[1]}) != model.nq ({model.nq})",
              file=sys.stderr)

    # Camera ID (or -1 for free camera using <global azimuth/elevation>)
    cam_id = -1
    if args.camera:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
        if cam_id < 0:
            print(f"[record] camera '{args.camera}' not found; using free camera",
                  file=sys.stderr)

    print(f"[record] creating offscreen Renderer {args.width}x{args.height}")
    renderer = mujoco.Renderer(model, args.height, args.width)

    # Free-camera defaults from the scene XML's <global>; if cam_id < 0, we
    # configure mjvCamera explicitly so the framing matches the Simulate viewer.
    free_cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, free_cam)
    # Pull back further so the whole body is in frame even when the root moves.
    free_cam.distance *= 1.3

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, args.fps,
                             (args.width, args.height))
    if not writer.isOpened():
        print(f"ERROR: failed to open VideoWriter for {args.output}", file=sys.stderr)
        return 3

    t0 = time.monotonic()
    for i, qpos in enumerate(motion):
        data.qpos[:] = qpos
        # Forward kinematics only; no physics integration.
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)  # so the camera tracking math works

        if cam_id >= 0:
            renderer.update_scene(data, camera=cam_id)
        else:
            renderer.update_scene(data, camera=free_cam)
        frame_rgb = renderer.render()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

        if i % 100 == 0:
            print(f"[record] frame {i}/{len(motion)}  ({100.0*i/len(motion):5.1f}%)")

    writer.release()
    renderer.close()
    dt = time.monotonic() - t0
    print(f"[record] done in {dt:.1f}s — wrote {len(motion)} frames to {args.output}")
    print(f"[record] effective render rate: {len(motion)/dt:.1f} fps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
