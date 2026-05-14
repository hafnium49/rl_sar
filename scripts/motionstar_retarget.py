"""Retarget Motion Star 15-sensor stream onto a Unitree G1 via GMR's mink+mujoco IK pipeline.

Reads an NPZ produced by fbx_motionstar_to_npz.py, feeds per-frame {sensor_name: (pos, quat)} dicts
into GMR's GeneralMotionRetargeting, and writes per-frame G1 qpos to a numpy archive (+ a JSON
companion with metadata).

Usage:
    /home/h_fujiwara/miniconda3/bin/conda run -n gmr python scripts/motionstar_retarget.py \\
        --npz <input.npz> --out <output_qpos.npz> [--max-frames N]

The output qpos array has shape (T, nq) where nq is G1's MuJoCo configuration dimension
(typically 36 = 3 root pos + 4 root quat + 29 joint angles for g1_mocap_29dof).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from general_motion_retargeting import GeneralMotionRetargeting

parser = argparse.ArgumentParser()
parser.add_argument("--npz", required=True, help="output of fbx_motionstar_to_npz.py")
parser.add_argument("--out", required=True, help="output .npz of per-frame G1 qpos + meta")
parser.add_argument("--human-height", type=float, default=1.45,
                    help="performer height in meters (Motion Star at Ibaraki ~ 1.45 m from rest pose)")
parser.add_argument("--max-frames", type=int, default=None,
                    help="truncate to first N frames (useful for smoke tests)")
parser.add_argument("--start-frame", type=int, default=1,
                    help="skip the first N frames (frame 0 of the FBX is sometimes empty)")
args = parser.parse_args()

data = np.load(args.npz)
pos_all = data["pos"]            # (T, 15, 3)
quat_all = data["quat"]          # (T, 15, 4)  wxyz
names = list(data["sensor_names"])
frame_time = float(data["frame_time"])

start = max(0, args.start_frame)
end = pos_all.shape[0] if args.max_frames is None else min(pos_all.shape[0], start + args.max_frames)
pos = pos_all[start:end]
quat = quat_all[start:end]
T = pos.shape[0]
print(f"[retarget] frames {start}..{end}  T={T}  frame_time={frame_time:.4f}")

retargeter = GeneralMotionRetargeting(
    src_human="motionstar",
    tgt_robot="unitree_g1",
    actual_human_height=args.human_height,
    verbose=False,
)
nq = retargeter.model.nq
print(f"[retarget] G1 model nq={nq}")

qpos_seq = np.zeros((T, nq), dtype=np.float32)

# GMR's offset_to_ground searches for body names containing "foot"/"Foot"; ours are
# "Sensor1".."Sensor15" so it would yield inf->NaN. Pre-ground here once instead:
# subtract the minimum sensor Z over the first frame's feet so they sit at +0.02 m.
foot_idx_left = names.index("Sensor1")
foot_idx_right = names.index("Sensor2")
floor_z = min(pos[0, foot_idx_left, 2], pos[0, foot_idx_right, 2]) - 0.02
pos[..., 2] -= floor_z
print(f"[retarget] grounded: subtracted floor_z={floor_z:.3f} m from all sensor Z values")

for f in tqdm(range(T), desc="retargeting"):
    human_data = {
        names[i]: (pos[f, i].astype(np.float64), quat[f, i].astype(np.float64))
        for i in range(len(names))
    }
    qpos = retargeter.retarget(human_data, offset_to_ground=False)
    qpos_seq[f] = qpos.astype(np.float32)

out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    out_path,
    qpos=qpos_seq,
    frame_time=np.float32(frame_time),
    n_frames=np.int32(T),
)
# Side-car metadata so the downstream CSV emitter knows the source
meta = {
    "source_npz": str(Path(args.npz).resolve()),
    "robot": "unitree_g1",
    "src_human": "motionstar",
    "frame_time": frame_time,
    "n_frames": T,
    "nq": int(nq),
    "human_height": args.human_height,
}
out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
print(f"[retarget] saved {out_path}  qpos shape: {qpos_seq.shape}")
