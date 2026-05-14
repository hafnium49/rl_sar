"""Headless FBX → BVH conversion using Blender's bpy API.

Invocation:
    blender --background --python scripts/fbx_to_bvh_blender.py -- <input.fbx> <output.bvh>

Used to feed GMR's bvh_to_robot.py route when the FBX SDK isn't available
(e.g. aarch64 Linux where AutoDesk's Python bindings aren't shipped).
"""
import sys

import bpy

# Parse args after the literal "--" separator (Blender convention)
try:
    sep = sys.argv.index("--")
except ValueError:
    print("ERROR: pass args after '--', e.g. blender -b -P this.py -- in.fbx out.bvh",
          file=sys.stderr)
    sys.exit(2)

args = sys.argv[sep + 1:]
if len(args) != 2:
    print(f"ERROR: expected 2 args (input.fbx output.bvh), got {len(args)}: {args}",
          file=sys.stderr)
    sys.exit(2)

input_fbx, output_bvh = args

# Clean default scene (Blender starts with a cube, a camera, a light)
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Import the FBX. import_anim=True so animation curves come along.
print(f"[fbx2bvh] importing {input_fbx}")
bpy.ops.import_scene.fbx(filepath=input_fbx, automatic_bone_orientation=True)

# Find the armature. There's usually exactly one in an Ibaraki mocap FBX.
armature = next((o for o in bpy.data.objects if o.type == 'ARMATURE'), None)
if armature is None:
    print("ERROR: no armature found after FBX import", file=sys.stderr)
    sys.exit(3)

print(f"[fbx2bvh] armature: {armature.name}  bones: {len(armature.data.bones)}  "
      f"frames: {bpy.context.scene.frame_end - bpy.context.scene.frame_start + 1}")

# Select armature so the BVH exporter targets it
bpy.ops.object.select_all(action='DESELECT')
armature.select_set(True)
bpy.context.view_layer.objects.active = armature

# Export BVH. global_scale=1.0 keeps source units; root_transform_only=False
# captures both root translation and rotation per frame; rotate_mode='NATIVE'
# preserves the source rotation order (FBX often uses Euler XYZ).
print(f"[fbx2bvh] exporting {output_bvh}")
bpy.ops.export_anim.bvh(
    filepath=output_bvh,
    global_scale=1.0,
    frame_start=bpy.context.scene.frame_start,
    frame_end=bpy.context.scene.frame_end,
    rotate_mode='NATIVE',
    root_transform_only=False,
)
print(f"[fbx2bvh] done: {output_bvh}")
