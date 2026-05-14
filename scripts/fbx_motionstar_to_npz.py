"""Extract per-frame world transforms of the 15 active Motion Star sensors from a converted-FBX.

Invocation:
    blender --background --python scripts/fbx_motionstar_to_npz.py -- <input.fbx> <output.npz>

The converted FBX (file-format 7700, from data/ibaraki_radio_taiso/converted/) contains 32 sensor
mesh empties parented under "MotionStar:Root". Only Sensor1..Sensor15 carry keyframes; the rest
are placeholders (Motion Star's file layout reserves 32 slots even when fewer sensors were wired).

Coordinate frame: source is Y-up centimeters. We rotate +90° about X (Y→Z, Z→-Y) and divide by 100
so the output is Z-up meters — what GMR's mink+mujoco pipeline expects for Unitree G1.
"""
import sys
import bpy
import numpy as np
from mathutils import Matrix, Quaternion

sep = sys.argv.index("--")
args = sys.argv[sep + 1:]
if len(args) != 2:
    print(f"ERROR: expected 2 args (input.fbx output.npz), got {len(args)}: {args}", file=sys.stderr)
    sys.exit(2)

input_fbx, output_npz = args

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

print(f"[motionstar] importing {input_fbx}")
bpy.ops.import_scene.fbx(filepath=input_fbx, automatic_bone_orientation=True)


def sensor_n(o):
    s = o.name.replace("MotionStar:Sensor", "")
    return int(s) if s.isdigit() else -1


all_sensors = sorted(
    [o for o in bpy.data.objects if o.name.startswith("MotionStar:Sensor")],
    key=sensor_n,
)
active_sensors = [
    s for s in all_sensors
    if s.animation_data and s.animation_data.action
    and any(len(fc.keyframe_points) > 0 for fc in s.animation_data.action.fcurves)
]
if len(active_sensors) != 15:
    print(f"ERROR: expected 15 active sensors, got {len(active_sensors)}", file=sys.stderr)
    sys.exit(3)

sensor_names = [s.name.replace("MotionStar:", "") for s in active_sensors]
print(f"[motionstar] active sensors: {sensor_names}")

# Animation length: read the maximum keyframe time across all sensor fcurves.
max_frame = 0
for s in active_sensors:
    for fc in s.animation_data.action.fcurves:
        if fc.keyframe_points:
            max_frame = max(max_frame, int(round(fc.keyframe_points[-1].co.x)))
n_frames = max_frame
print(f"[motionstar] n_frames={n_frames}  duration_s={n_frames / 30.0:.1f}")

# Extend scene range so dependency graph evaluates correctly when scrubbing.
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = n_frames

# Y-up → Z-up rotation: +90° about X. (x,y,z)_yup → (x, -z, y)_zup.
R_y2z = Matrix.Rotation(1.5707963267948966, 4, 'X')


def collect_fcurves(obj):
    """Index this object's fcurves by (data_path, array_index) so per-frame eval is O(1)."""
    if not (obj.animation_data and obj.animation_data.action):
        return None
    fcs = {}
    for fc in obj.animation_data.action.fcurves:
        fcs[(fc.data_path, fc.array_index)] = fc
    return fcs


def eval_matrix(fcs, rotation_mode, frame):
    """Reconstruct an object's local-to-parent matrix from its fcurves at the given frame."""
    if fcs is None:
        return Matrix.Identity(4)
    lx = fcs.get(("location", 0))
    ly = fcs.get(("location", 1))
    lz = fcs.get(("location", 2))
    loc = (
        lx.evaluate(frame) if lx else 0.0,
        ly.evaluate(frame) if ly else 0.0,
        lz.evaluate(frame) if lz else 0.0,
    )
    if rotation_mode == 'QUATERNION':
        qw = fcs.get(("rotation_quaternion", 0))
        qx = fcs.get(("rotation_quaternion", 1))
        qy = fcs.get(("rotation_quaternion", 2))
        qz = fcs.get(("rotation_quaternion", 3))
        q = Quaternion((
            qw.evaluate(frame) if qw else 1.0,
            qx.evaluate(frame) if qx else 0.0,
            qy.evaluate(frame) if qy else 0.0,
            qz.evaluate(frame) if qz else 0.0,
        ))
        rot_mat = q.to_matrix().to_4x4()
    else:
        # default rotation_mode='XYZ' (intrinsic euler)
        from mathutils import Euler
        rx = fcs.get(("rotation_euler", 0))
        ry = fcs.get(("rotation_euler", 1))
        rz = fcs.get(("rotation_euler", 2))
        e = Euler((
            rx.evaluate(frame) if rx else 0.0,
            ry.evaluate(frame) if ry else 0.0,
            rz.evaluate(frame) if rz else 0.0,
        ), rotation_mode or 'XYZ')
        rot_mat = e.to_matrix().to_4x4()
    trans_mat = Matrix.Translation(loc)
    return trans_mat @ rot_mat


# Cache the root and per-sensor fcurves once; check whether root is identity (skip if so).
root_obj = next((o for o in bpy.data.objects if o.name == "MotionStar:Root"), None)
root_fcs = collect_fcurves(root_obj) if root_obj else None
root_mode = root_obj.rotation_mode if root_obj else 'XYZ'
sensor_fcs = [collect_fcurves(s) for s in active_sensors]
sensor_modes = [s.rotation_mode for s in active_sensors]
print(f"[motionstar] root rotation_mode={root_mode}  sensor rotation_mode={sensor_modes[0]}")

pos = np.zeros((n_frames, 15, 3), dtype=np.float32)
quat = np.zeros((n_frames, 15, 4), dtype=np.float32)

for f_idx in range(n_frames):
    # Keyframes start at frame 2 (per inspection), so source-frame = output_idx + 2 is safer
    # but we'll use 1-based frame numbers directly — fcurve.evaluate handles the offset.
    f = f_idx + 1  # 1-based for fcurve eval
    M_root = eval_matrix(root_fcs, root_mode, f) if root_fcs else Matrix.Identity(4)
    for i, s in enumerate(active_sensors):
        M_sensor_local = eval_matrix(sensor_fcs[i], sensor_modes[i], f)
        M_world_yup = M_root @ M_sensor_local
        M_world_zup = R_y2z @ M_world_yup
        t = M_world_zup.to_translation()
        q = M_world_zup.to_quaternion()
        pos[f_idx, i] = (t.x / 100.0, t.y / 100.0, t.z / 100.0)
        quat[f_idx, i] = (q.w, q.x, q.y, q.z)
    if f_idx % 500 == 0:
        print(f"  frame {f}/{n_frames}  (pelvis Z = {pos[f_idx, 6, 2]:.2f} m)")

print(f"[motionstar] frame 1: head Z = {pos[0, 14, 2]:.2f} m  (should be ~1.5 — sanity)")
print(f"[motionstar] frame 1: left_foot Z = {pos[0, 0, 2]:.2f} m  (should be near 0)")

np.savez_compressed(
    output_npz,
    pos=pos,
    quat=quat,
    frame_time=np.float32(1.0 / 30.0),
    n_frames=np.int32(n_frames),
    sensor_names=np.array(sensor_names),
)
print(f"[motionstar] saved {output_npz}  ({pos.nbytes + quat.nbytes:,} bytes raw)")
