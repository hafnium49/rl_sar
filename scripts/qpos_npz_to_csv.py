"""Convert a retargeted qpos NPZ (MuJoCo wxyz order) to the 36-column CSV the rl_sar renderer expects.

CSV layout (per row, matches policy/g1/whole_body_tracking/dance_102/G1_Take_102.bvh_60hz.csv):
    [0,1,2]   root_pos x y z
    [3,4,5,6] root_quat x y z w     (the renderer's load_motion() permutes back to wxyz)
    [7..35]   29 joint angles

MuJoCo qpos layout (what we have):
    [0,1,2]   root_pos x y z
    [3,4,5,6] root_quat w x y z
    [7..35]   29 joint angles

So we move qpos[3] (qw) to csv[6] and shift qpos[4:7] (qxyz) down to csv[3:6].
"""
import argparse
import csv
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--qpos", required=True, help="output of motionstar_retarget.py")
ap.add_argument("--csv", required=True, help="output CSV path")
args = ap.parse_args()

d = np.load(args.qpos)
qpos = d["qpos"]  # (T, 36)
T, nq = qpos.shape
if nq != 36:
    raise SystemExit(f"expected nq=36, got {nq}")

out = np.empty_like(qpos)
out[:, 0:3] = qpos[:, 0:3]        # root pos
out[:, 3:6] = qpos[:, 4:7]        # qx qy qz
out[:, 6]   = qpos[:, 3]          # qw  → col 6 (xyzw order)
out[:, 7:]  = qpos[:, 7:]         # joints

Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
with open(args.csv, "w", newline="") as f:
    w = csv.writer(f)
    for row in out:
        w.writerow([f"{x:.6f}" for x in row])
print(f"wrote {args.csv}  rows={T}  cols={nq}")
