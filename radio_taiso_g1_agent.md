# Agent: Build a Unitree G1 ラジオ体操 Reference-Motion and Physics-AI Training Pipeline

**Target reader:** Claude Code or another coding agent working in a Linux development environment.  
**Primary goal:** Create a reproducible pipeline that starts from public ラジオ体操 source motion data, retargets it to Unitree G1 using both GMR and NMR where possible, trains a Unitree G1 motion-tracking PPO controller, verifies it in MuJoCo, and prepares—but does not execute—real-robot deployment steps.

---

## 0. Mission Summary

There does **not** appear to be a ready-made public Unitree G1 ラジオ体操 RL policy or G1-format ラジオ体操 CSV file. The implementation must therefore build the pipeline:

```text
Radio Taiso source motion
→ GMR and/or NMR retargeting
→ Unitree G1 reference motion
→ unitree_rl_mjlab tracking PPO
→ MuJoCo play / sim2real verification
→ optional physical deployment preparation
```

Two retargeting routes must be supported:

1. **Route A — GMR:** `YanjieZe/GMR`  
   Repository: https://github.com/YanjieZe/GMR

2. **Route B — NMR / MakeTrackingEasy:** `NJU3DV-HumanoidGroup/MakeTrackingEasy`  
   Repository: https://github.com/NJU3DV-HumanoidGroup/MakeTrackingEasy

Downstream training route:

3. **Unitree RL Mjlab:** `unitreerobotics/unitree_rl_mjlab`  
   Repository: https://github.com/unitreerobotics/unitree_rl_mjlab

Optional simulation/deployment reference:

4. **Unitree MuJoCo:** `unitreerobotics/unitree_mujoco`  
   Repository: https://github.com/unitreerobotics/unitree_mujoco

---

## 1. Non-Negotiable Constraints

1. **Start from source data download.** Do not skip the source-data acquisition step.
2. **Do not assume a ready-made Unitree G1 Radio Taiso CSV or policy exists.** Search if needed, but implement the pipeline as if no ready-made file exists.
3. **Do not deploy to physical Unitree G1.** Only prepare the physical-deployment folder and commands. Real hardware execution requires explicit human approval.
4. **Retain both retargeting routes.** GMR is the first route; NMR is a second benchmark route if SMPL-X / AMASS-style input can be produced.
5. **Do not hardcode `take05` as the only Radio Taiso source.** Public search results have shown several Ibaraki MoCap Radio Taiso pages/takes. Download and compare available takes.
6. **All generated artifacts must be versioned and documented:** source file, conversion route, retargeter, FPS, robot target, output path, and validation result.
7. **Every command must be made robust to path differences.** Use `find`, `--help`, and README inspection when exact script arguments differ.
8. **Simulation gate comes before hardware gate.** The policy must complete repeated full-sequence MuJoCo simulations without falling, joint-limit hammering, severe foot skating, uncontrolled yaw drift, or high-torque spikes before real-robot deployment is even considered.

---

## 2. Recommended Repository Layout

Use this working layout unless the user specifies another root:

```text
~/projects/
  GMR/
  MakeTrackingEasy/
  unitree_rl_mjlab/
  unitree_mujoco/                # optional

~/datasets/radio_taiso/
  source/
    ibaraki_fbx/
    mmd_vmd_fallback/
  intermediate/
  gmr/
  gmr_csv/
  nmr_input/
  nmr_output/
  nmr_csv/
  reports/
  videos/
  logs/
```

Create directories:

```bash
mkdir -p ~/projects
mkdir -p ~/datasets/radio_taiso/{source/ibaraki_fbx,source/mmd_vmd_fallback,intermediate,gmr,gmr_csv,nmr_input,nmr_output,nmr_csv,reports,videos,logs}
```

---

## 3. Source Data URLs

### 3.1 Primary Source: Ibaraki University MoCap Archive

Use Ibaraki University MoCap as the first source because FBX is easier to convert than MMD/VMD.

Known Radio Taiso related pages from previous exploration:

- https://fondant.cis.ibaraki.ac.jp/MoCap/0913.htm
- https://fondant.cis.ibaraki.ac.jp/MoCap/0912.htm
- https://fondant.cis.ibaraki.ac.jp/MoCap/0207.htm

Expected labels to look for:

```text
ラジオ体操第一
Radio Taiso First
Radio exercise
```

Expected downloadable formats may include:

```text
*.zip containing *.fbx
*.zip containing *.vpm
movie files
```

Known examples from prior review:

```text
take06.zip (fbx)
take09.zip (fbx)
take10.zip (fbx)
```

Do **not** assume the exact available take numbers. Inspect the page manually or with a downloader and record what is actually available.

Manual download is acceptable. Put all downloaded zip files here:

```text
~/datasets/radio_taiso/source/ibaraki_fbx/
```

Unzip:

```bash
cd ~/datasets/radio_taiso/source/ibaraki_fbx

# Replace with the actual downloaded files.
unzip take06.zip -d take06_fbx || true
unzip take09.zip -d take09_fbx || true
unzip take10.zip -d take10_fbx || true

find ~/datasets/radio_taiso/source/ibaraki_fbx -type f | sort | tee ~/datasets/radio_taiso/reports/source_files.txt
```

Select the best take using these criteria:

```text
1. Full-body motion continuity
2. No missing frames
3. Correct floor contact
4. No unnatural root drift
5. Good arm-swing quality
6. Clear start and end standing pose
```

Document selected source:

```bash
cat > ~/datasets/radio_taiso/reports/source_selection.md <<'EOF_SOURCE'
# Radio Taiso Source Selection

## Candidate files

TODO: list downloaded files.

## Selected source

TODO: path to selected FBX/BVH/SMPL-X/AMASS file.

## Reason

TODO: explain why selected take was chosen.
EOF_SOURCE
```

### 3.2 Fallback Source: MMD/VMD Motions

If Ibaraki FBX cannot be processed, use MMD/VMD Radio Taiso sources as fallback. These are not robot-ready and may require Blender/MMD conversion.

Known sources from prior review:

- BowlRoll: `モーション_ラジオ体操第一`  
  https://bowlroll.net/file/303669

- BowlRoll: `真面目なラジオ体操モーション`  
  https://bowlroll.net/file/100945

- BowlRoll: `ラジオ体操第一セット Ver.3`  
  https://bowlroll.net/file/304187

- VPVP wiki / MMD motion index  
  https://w.atwiki.jp/vpvpwiki/pages/236.html

Useful conversion tools:

- Blender MMD Tools: https://github.com/powroupi/blender_mmd_tools
- PV2FC / PMX-VMD-to-FBX type route, if needed: inspect current public tools before use.

Fallback conversion concept:

```text
VMD/MMD motion
→ Blender import using mmd_tools
→ export as FBX or BVH
→ GMR route
```

---

# Route A: Retarget Using GMR

Use GMR first. It is the most practical first-choice retargeter for Unitree G1 because it directly supports multiple source formats and robot targets.

GMR repository:

```text
https://github.com/YanjieZe/GMR
```

Important GMR robot targets:

```text
unitree_g1              # standard Unitree G1 body, 29 DoF
unitree_g1_with_hands   # Unitree G1 body with hands, 43 DoF
```

For Radio Taiso, use `unitree_g1` first. Radio Taiso does not require dexterous finger motion, and the downstream `unitree_rl_mjlab` G1 tracking route is more direct with the standard G1 body.

---

## 4A. Clone and Install GMR

```bash
cd ~/projects
git clone https://github.com/YanjieZe/GMR.git
cd GMR

conda create -n gmr python=3.10 -y
conda activate gmr

pip install -e .
```

If using FBX, inspect GMR README for FBX SDK requirements. GMR’s documented FBX route uses:

```text
third_party/poselib/fbx_importer.py
scripts/fbx_offline_to_robot.py
```

Check available scripts:

```bash
cd ~/projects/GMR
find scripts third_party -maxdepth 3 -type f | sort | tee ~/datasets/radio_taiso/reports/gmr_script_inventory.txt
python scripts/fbx_offline_to_robot.py --help || true
python scripts/bvh_to_robot.py --help || true
python scripts/smplx_to_robot.py --help || true
python scripts/vis_robot_motion.py --help || true
```

---

## 5A. Extract Motion from FBX

Try the direct GMR FBX route first.

Create intermediate folder:

```bash
mkdir -p ~/datasets/radio_taiso/intermediate
```

Run FBX importer. Replace `<radio_taiso>.fbx` with the actual selected FBX path.

```bash
cd ~/projects/GMR/third_party
conda activate gmr

python poselib/fbx_importer.py \
  --input ~/datasets/radio_taiso/source/ibaraki_fbx/take06_fbx/<radio_taiso>.fbx \
  --output ~/datasets/radio_taiso/intermediate/radio_taiso_take06_fbx_motion.pkl \
  --root-joint Hips \
  --fps 30
```

If `--root-joint Hips` fails, inspect the FBX skeleton in Blender or an FBX viewer and replace the root joint with the actual name.

Record actual root joint and FPS in:

```text
~/datasets/radio_taiso/reports/source_selection.md
```

---

## 6A. Retarget FBX Motion to Unitree G1 Using GMR

Standard G1 route:

```bash
cd ~/projects/GMR
conda activate gmr

python scripts/fbx_offline_to_robot.py \
  --motion_file ~/datasets/radio_taiso/intermediate/radio_taiso_take06_fbx_motion.pkl \
  --robot unitree_g1 \
  --save_path ~/datasets/radio_taiso/gmr/radio_taiso_g1.pkl \
  --rate_limit
```

Optional G1-with-hands route, only for comparison:

```bash
python scripts/fbx_offline_to_robot.py \
  --motion_file ~/datasets/radio_taiso/intermediate/radio_taiso_take06_fbx_motion.pkl \
  --robot unitree_g1_with_hands \
  --save_path ~/datasets/radio_taiso/gmr/radio_taiso_g1_hands.pkl \
  --rate_limit
```

If the script opens MuJoCo visualization, visually inspect the output and record whether it is acceptable.

---

## 7A. If FBX Fails, Convert FBX to BVH or SMPL-X and Retry

GMR’s FBX route may assume OptiTrack-style FBX conventions. Ibaraki FBX may use a different skeleton. If direct FBX fails, use one of these routes:

```text
FBX → BVH → GMR bvh_to_robot.py
FBX → SMPL-X / AMASS-style motion → GMR smplx_to_robot.py
```

### BVH route

Convert FBX to BVH using Blender or another conversion tool. Then:

```bash
cd ~/projects/GMR
conda activate gmr

python scripts/bvh_to_robot.py \
  --bvh_file ~/datasets/radio_taiso/intermediate/radio_taiso_take06.bvh \
  --robot unitree_g1 \
  --save_path ~/datasets/radio_taiso/gmr/radio_taiso_g1.pkl \
  --rate_limit \
  --format lafan1
```

If `--format lafan1` is wrong, inspect `python scripts/bvh_to_robot.py --help` and choose the closest supported format.

### SMPL-X route

If you can produce SMPL-X `.pkl`:

```bash
cd ~/projects/GMR
conda activate gmr

python scripts/smplx_to_robot.py \
  --smplx_file ~/datasets/radio_taiso/intermediate/radio_taiso_take06_smplx.pkl \
  --robot unitree_g1 \
  --save_path ~/datasets/radio_taiso/gmr/radio_taiso_g1.pkl \
  --rate_limit \
  --record_video \
  --video_path ~/datasets/radio_taiso/gmr/radio_taiso_g1_preview.mp4
```

---

## 8A. Visualize GMR Retargeted Motion

```bash
cd ~/projects/GMR
conda activate gmr

python scripts/vis_robot_motion.py \
  --robot unitree_g1 \
  --robot_motion_path ~/datasets/radio_taiso/gmr/radio_taiso_g1.pkl \
  --record_video \
  --video_path ~/datasets/radio_taiso/gmr/radio_taiso_g1_preview.mp4
```

Inspect:

```text
1. Feet do not slide excessively
2. Knees do not hyperextend
3. Torso remains plausible
4. Arms do not self-collide
5. No sudden shoulder or waist jumps
6. Motion starts from a safe standing pose
7. Motion ends in a recoverable standing pose
```

Record result:

```bash
cat > ~/datasets/radio_taiso/reports/gmr_retarget_report.md <<'EOF_GMR'
# GMR Retarget Report

## Input
TODO

## Robot target
unitree_g1

## Output
~/datasets/radio_taiso/gmr/radio_taiso_g1.pkl

## Visual inspection
- Feet sliding: TODO
- Joint jumps: TODO
- Self-collision: TODO
- Start/end safety: TODO

## Decision
TODO: ACCEPT / REJECT / NEEDS TUNING
EOF_GMR
```

---

## 9A. Convert GMR Output to CSV for unitree_rl_mjlab

GMR commonly outputs robot motion as `.pkl`. Convert it to CSV.

Try GMR’s batch conversion utility if present:

```bash
cd ~/projects/GMR
conda activate gmr

python scripts/batch_gmr_pkl_to_csv.py \
  --src_folder ~/datasets/radio_taiso/gmr \
  --tgt_folder ~/datasets/radio_taiso/gmr_csv \
  --robot unitree_g1
```

Expected output:

```text
~/datasets/radio_taiso/gmr_csv/radio_taiso_g1.csv
```

If the script or arguments differ:

```bash
cd ~/projects/GMR
find scripts -iname "*csv*" -o -iname "*pkl*"
python scripts/batch_gmr_pkl_to_csv.py --help || true
```

Do not proceed blindly. Match the converter to the local checkout.

---

# Route B: Retarget Using NMR / MakeTrackingEasy

Use NMR as a second benchmark route. Do not block the first sprint on NMR if converting the source to SMPL-X / AMASS-style `.npz` is difficult.

NMR repository:

```text
https://github.com/NJU3DV-HumanoidGroup/MakeTrackingEasy
```

NMR paper/project name:

```text
NMR: Neural Motion Retargeting for Humanoid Whole-body Control
```

Current public-release expectations from prior review:

```text
- Inference code and checkpoints are available.
- Training code and CEPR dataset may still be TODO / incomplete.
- Input: AMASS .npz or standard SMPL-X-like .npz.
- Output: bmimic .npz at 50 FPS with G1 joint_pos shape (T, 29).
```

---

## 4B. Clone and Install NMR

```bash
cd ~/projects
git clone https://github.com/NJU3DV-HumanoidGroup/MakeTrackingEasy.git
cd MakeTrackingEasy

conda create -n nmr python=3.10 -y
conda activate nmr

pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Inspect available CLI:

```bash
cd ~/projects/MakeTrackingEasy
conda activate nmr

python inference.py --help || true
python visualize.py --help || true
find . -maxdepth 3 -type f | sort | tee ~/datasets/radio_taiso/reports/nmr_script_inventory.txt
```

---

## 5B. Convert Radio Taiso Source Motion into NMR Input Format

NMR does **not** directly consume arbitrary FBX. It expects either AMASS-style `.npz` or standard SMPL-X-like `.npz`.

Expected AMASS-style fields:

```text
trans
root_orient
pose_body
```

Expected standard `.npz` fields:

```text
transl
global_orient
body_pose
```

Required concept:

```text
Ibaraki FBX
→ BVH / SMPL-X / AMASS-style .npz
→ NMR inference
→ G1 bmimic .npz
```

Create input directory:

```bash
mkdir -p ~/datasets/radio_taiso/nmr_input
```

Expected target after conversion:

```text
~/datasets/radio_taiso/nmr_input/radio_taiso_take06_amass.npz
```

If you cannot produce this `.npz`, skip NMR for the first sprint and continue with GMR.

Record blocker if any:

```bash
cat > ~/datasets/radio_taiso/reports/nmr_input_conversion_status.md <<'EOF_NMR_INPUT'
# NMR Input Conversion Status

## Goal
Convert Radio Taiso source motion to AMASS / SMPL-X-style .npz.

## Expected fields
- trans or transl
- root_orient or global_orient
- pose_body or body_pose

## Status
TODO: SUCCESS / BLOCKED

## Blocker details
TODO
EOF_NMR_INPUT
```

---

## 6B. Run NMR Inference

```bash
cd ~/projects/MakeTrackingEasy
conda activate nmr

python inference.py \
  --src ~/datasets/radio_taiso/nmr_input/radio_taiso_take06_amass.npz \
  --output-dir ~/datasets/radio_taiso/nmr_output
```

For batch processing, inspect README or `python inference.py --help` and use directory input if supported.

Expected output:

```text
~/datasets/radio_taiso/nmr_output/*.npz
```

Expected output structure:

```text
fps
joint_pos      # (T, 29), G1 joint angles
joint_vel      # (T, 29)
body_pos_w
body_quat_w
body_lin_vel_w
```

Inspect output:

```bash
python - <<'PY'
import numpy as np
from pathlib import Path
for p in sorted(Path('~/datasets/radio_taiso/nmr_output').expanduser().glob('*.npz')):
    print('FILE', p)
    data = np.load(p, allow_pickle=True)
    print('keys:', sorted(data.files))
    for k in data.files:
        v = data[k]
        try:
            print(' ', k, v.shape, v.dtype)
        except Exception:
            print(' ', k, type(v))
PY
```

---

## 7B. Visualize NMR Output

```bash
cd ~/projects/MakeTrackingEasy
conda activate nmr

python visualize.py \
  --src ~/datasets/radio_taiso/nmr_output/<radio_taiso_nmr_output>.npz
```

If flag names differ:

```bash
python visualize.py --help
```

Inspect:

```text
1. Foot stability
2. No sudden joint jumps
3. No severe self-collision
4. Torso and arms preserve Radio Taiso character
5. Start and end are recoverable
```

Record result:

```bash
cat > ~/datasets/radio_taiso/reports/nmr_retarget_report.md <<'EOF_NMR'
# NMR Retarget Report

## Input
TODO

## Output
TODO

## Visual inspection
- Feet sliding: TODO
- Joint jumps: TODO
- Self-collision: TODO
- Start/end safety: TODO

## Decision
TODO: ACCEPT / REJECT / NEEDS TUNING
EOF_NMR
```

---

## 8B. Convert NMR bmimic `.npz` to unitree_rl_mjlab CSV

NMR output already contains G1 `joint_pos: (T, 29)` and likely 50 FPS. `unitree_rl_mjlab` expects a CSV that can be passed into `scripts/csv_to_npz.py`.

Create converter:

```bash
mkdir -p ~/datasets/radio_taiso/nmr_csv
mkdir -p ~/projects/radio_taiso_tools
cat > ~/projects/radio_taiso_tools/nmr_bmimic_to_unitree_csv.py <<'PY'
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='NMR bmimic npz')
    parser.add_argument('--output', required=True, help='Output CSV for unitree_rl_mjlab')
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=True)
    if 'joint_pos' not in data.files:
        raise KeyError(f"Expected key 'joint_pos'. Available keys: {data.files}")

    joint_pos = data['joint_pos']
    if joint_pos.ndim != 2:
        raise ValueError(f'joint_pos must be 2D, got shape {joint_pos.shape}')
    if joint_pos.shape[1] != 29:
        raise ValueError(f'Expected G1 29-DoF joint_pos shape (T,29), got {joint_pos.shape}')

    # IMPORTANT: This header is a placeholder. Before training, compare with
    # unitree_rl_mjlab's example CSV and replace column names/order if needed.
    columns = [f'joint_{i}' for i in range(joint_pos.shape[1])]
    df = pd.DataFrame(joint_pos, columns=columns)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f'Wrote {out} with shape {df.shape}')


if __name__ == '__main__':
    main()
PY
```

Run converter:

```bash
python ~/projects/radio_taiso_tools/nmr_bmimic_to_unitree_csv.py \
  --input ~/datasets/radio_taiso/nmr_output/<radio_taiso_nmr_output>.npz \
  --output ~/datasets/radio_taiso/nmr_csv/radio_taiso_nmr_g1.csv
```

Before training, compare against Unitree’s example CSV:

```bash
cd ~/projects/unitree_rl_mjlab || true
head -5 src/assets/motions/g1/dance1_subject2.csv || true
head -5 ~/datasets/radio_taiso/nmr_csv/radio_taiso_nmr_g1.csv
```

If headers/order differ, update `nmr_bmimic_to_unitree_csv.py` to match the actual `dance1_subject2.csv` convention.

---

# Phase 2: RL Motion-Tracking Training with unitree_rl_mjlab

Unitree RL Mjlab repository:

```text
https://github.com/unitreerobotics/unitree_rl_mjlab
```

Workflow:

```text
Train → Play → Sim2Real
```

---

## 10. Clone and Prepare unitree_rl_mjlab

```bash
cd ~/projects
git clone https://github.com/unitreerobotics/unitree_rl_mjlab.git
cd unitree_rl_mjlab
```

Follow repository setup instructions:

```bash
# Inspect setup docs.
find . -maxdepth 3 -iname '*setup*' -o -iname 'README*'
cat README.md | head -120

# Follow doc/setup.md or the current README instructions.
```

Create motion directory:

```bash
mkdir -p src/assets/motions/g1
```

Copy GMR CSV:

```bash
cp ~/datasets/radio_taiso/gmr_csv/radio_taiso_g1.csv \
   src/assets/motions/g1/radio_taiso_gmr_g1.csv
```

Copy NMR CSV, if available:

```bash
cp ~/datasets/radio_taiso/nmr_csv/radio_taiso_nmr_g1.csv \
   src/assets/motions/g1/radio_taiso_nmr_g1.csv || true
```

---

## 11. Convert CSV to NPZ

GMR route, assuming source is 30 FPS:

```bash
cd ~/projects/unitree_rl_mjlab

python scripts/csv_to_npz.py \
  --input-file src/assets/motions/g1/radio_taiso_gmr_g1.csv \
  --output-name radio_taiso_gmr_g1.npz \
  --input-fps 30 \
  --output-fps 50 \
  --robot g1
```

NMR route, assuming output is already 50 FPS:

```bash
python scripts/csv_to_npz.py \
  --input-file src/assets/motions/g1/radio_taiso_nmr_g1.csv \
  --output-name radio_taiso_nmr_g1.npz \
  --input-fps 50 \
  --output-fps 50 \
  --robot g1
```

Find actual output path because README path conventions may vary:

```bash
find . -name 'radio_taiso*.npz' | sort | tee ~/datasets/radio_taiso/reports/unitree_npz_paths.txt
```

Set variables for later commands:

```bash
export RADIO_TAISO_GMR_NPZ=$(find . -name 'radio_taiso_gmr_g1.npz' | head -1)
export RADIO_TAISO_NMR_NPZ=$(find . -name 'radio_taiso_nmr_g1.npz' | head -1)
echo "GMR NPZ=$RADIO_TAISO_GMR_NPZ"
echo "NMR NPZ=$RADIO_TAISO_NMR_NPZ"
```

---

## 12. Train PPO Tracking Policy

GMR-retargeted motion:

```bash
cd ~/projects/unitree_rl_mjlab

python scripts/train.py \
  Unitree-G1-Tracking-No-State-Estimation \
  --motion_file="$RADIO_TAISO_GMR_NPZ" \
  --env.scene.num-envs=4096
```

NMR-retargeted motion, if available:

```bash
python scripts/train.py \
  Unitree-G1-Tracking-No-State-Estimation \
  --motion_file="$RADIO_TAISO_NMR_NPZ" \
  --env.scene.num-envs=4096
```

Multi-GPU, if available:

```bash
python scripts/train.py \
  Unitree-G1-Tracking-No-State-Estimation \
  --motion_file="$RADIO_TAISO_GMR_NPZ" \
  --env.scene.num-envs=4096 \
  --gpu-ids 0 1
```

Record run paths:

```bash
find logs/rsl_rl -name 'model_*.pt' | sort | tail -20 | tee ~/datasets/radio_taiso/reports/training_checkpoints.txt
find logs/rsl_rl -name 'policy.onnx*' | sort | tee ~/datasets/radio_taiso/reports/exported_onnx.txt
```

Expected output structure:

```text
logs/rsl_rl/g1_tracking/<date_time>/model_<iteration>.pt
logs/rsl_rl/g1_tracking/<date_time>/policy.onnx
logs/rsl_rl/g1_tracking/<date_time>/policy.onnx.data
```

---

# Phase 3: Deployment and Verification in MuJoCo

---

## 13. Play / Verify in MuJoCo

GMR-trained policy:

```bash
cd ~/projects/unitree_rl_mjlab

python scripts/play.py \
  Unitree-G1-Tracking-No-State-Estimation \
  --motion_file="$RADIO_TAISO_GMR_NPZ" \
  --checkpoint_file=logs/rsl_rl/g1_tracking/<timestamp>/model_<iteration>.pt
```

NMR-trained policy:

```bash
python scripts/play.py \
  Unitree-G1-Tracking-No-State-Estimation \
  --motion_file="$RADIO_TAISO_NMR_NPZ" \
  --checkpoint_file=logs/rsl_rl/g1_tracking/<timestamp>/model_<iteration>.pt
```

Run multiple trials.

Simulation validation checklist:

```text
1. Full Radio Taiso sequence completes without falling
2. The policy survives deep arm arcs
3. The policy survives torso bending and twisting
4. No joint saturation
5. No repeated foot slip
6. No high-torque spikes
7. Start transition is safe
8. End transition is safe
9. No uncontrolled yaw drift
10. No severe visual mismatch from Radio Taiso character
```

Recommended simulation gate:

```text
20/20 full-sequence completions in MuJoCo
No falls
No visible joint-limit hammering
No severe foot skating
No uncontrolled yaw drift
No high-torque spikes
```

Write report:

```bash
cat > ~/datasets/radio_taiso/reports/mujoco_play_report.md <<'EOF_PLAY'
# MuJoCo Play Report

## GMR route
- Motion file: TODO
- Checkpoint: TODO
- Trials: TODO
- Full completions: TODO
- Falls: TODO
- Foot slip: TODO
- Joint-limit events: TODO
- Verdict: TODO

## NMR route
- Motion file: TODO
- Checkpoint: TODO
- Trials: TODO
- Full completions: TODO
- Falls: TODO
- Foot slip: TODO
- Joint-limit events: TODO
- Verdict: TODO

## Selected route
TODO: GMR / NMR / neither
EOF_PLAY
```

---

## 14. Compare GMR vs NMR

If both routes complete, compare:

```text
1. Tracking success rate
2. Fall rate
3. Foot slip
4. Joint velocity spikes
5. Torque spikes
6. Visual similarity to ラジオ体操
7. Smoothness of arms and waist
8. Ease of conversion into unitree_rl_mjlab
9. Training stability
10. Final checkpoint quality
```

Expected practical interpretation:

```text
GMR:
  More practical first route.
  Better source-format flexibility.
  Easier if starting from FBX/BVH.

NMR:
  Potentially smoother or more dynamically friendly if SMPL-X/AMASS input is clean.
  Harder if source data is only FBX.
  Newer codebase; training code/dataset release status may still be incomplete.
```

---

# Phase 4: Physical Deployment Preparation Only

Do **not** run on physical G1 unless explicitly approved.

Hard gate:

```text
Do not deploy to real G1 until the policy completes the full Radio Taiso sequence
in MuJoCo for many randomized trials without falling, saturating joints, or
producing high-torque spikes.
```

Unitree physical deployment route involves:

```text
cyclonedds
unitree_sdk2
Ethernet configuration
CMake compilation under deploy/robots/g1
Simulation deployment using unitree_mujoco first
Physical deployment only after simulation controller works
```

Prepare exported policy folder:

```bash
cd ~/projects/unitree_rl_mjlab
mkdir -p deploy/robots/g1/config/policy/tracking/v0/exported

cp logs/rsl_rl/g1_tracking/<timestamp>/policy.onnx \
   deploy/robots/g1/config/policy/tracking/v0/exported/

cp logs/rsl_rl/g1_tracking/<timestamp>/policy.onnx.data \
   deploy/robots/g1/config/policy/tracking/v0/exported/
```

Compile deployment controller:

```bash
cd ~/projects/unitree_rl_mjlab/deploy/robots/g1
mkdir -p build
cd build
cmake ..
make
```

Simulation deployment first:

```bash
cd ~/projects/unitree_rl_mjlab/simulate
mkdir -p build
cd build
cmake ..
make -j8

./unitree_mujoco
```

Launch G1 controller against simulation only:

```bash
cd ~/projects/unitree_rl_mjlab/deploy/robots/g1/build
./g1_ctrl --network=lo
```

Only after explicit human approval and after simulation succeeds:

```bash
./g1_ctrl --network=<your_real_ethernet_interface>
```

Do not execute the real-hardware command automatically.

---

# Recommended First Sprint

Execute exactly this order:

```text
1. Download Ibaraki ラジオ体操第一 FBX takes.
2. Inspect and choose the best take.
3. Clone and install GMR.
4. Retarget one take with GMR to unitree_g1.
5. Visualize GMR result in MuJoCo.
6. Convert GMR .pkl to CSV.
7. Clone and prepare unitree_rl_mjlab.
8. Convert CSV to NPZ using unitree_rl_mjlab.
9. Train Unitree-G1-Tracking-No-State-Estimation.
10. Play the trained policy in MuJoCo.
11. In parallel, prepare the NMR path only if SMPL-X/AMASS-style .npz can be produced.
12. Run NMR inference.
13. Convert NMR bmimic .npz to unitree_rl_mjlab CSV.
14. Train a second policy from NMR output.
15. Compare GMR-policy vs NMR-policy.
16. Prepare physical deployment files only after simulation is stable.
17. Do not run real hardware without explicit approval.
```

---

## Final Acceptance Criteria

A successful implementation produces:

```text
~/datasets/radio_taiso/reports/source_selection.md
~/datasets/radio_taiso/gmr/radio_taiso_g1.pkl
~/datasets/radio_taiso/gmr_csv/radio_taiso_g1.csv
~/datasets/radio_taiso/reports/gmr_retarget_report.md
~/projects/unitree_rl_mjlab/src/assets/motions/g1/radio_taiso_gmr_g1.csv
<actual path>/radio_taiso_gmr_g1.npz
logs/rsl_rl/g1_tracking/<timestamp>/model_<iteration>.pt
~/datasets/radio_taiso/reports/mujoco_play_report.md
```

Optional NMR outputs:

```text
~/datasets/radio_taiso/nmr_input/radio_taiso_take06_amass.npz
~/datasets/radio_taiso/nmr_output/<radio_taiso_nmr_output>.npz
~/datasets/radio_taiso/nmr_csv/radio_taiso_nmr_g1.csv
<actual path>/radio_taiso_nmr_g1.npz
~/datasets/radio_taiso/reports/nmr_retarget_report.md
```

Final report must state:

```text
1. Which source motion was used
2. Which retargeting route worked
3. Which route produced the best MuJoCo result
4. Whether full-sequence Radio Taiso was completed in simulation
5. Whether the policy is ready for further controlled testing
6. Whether physical deployment is still blocked
```

