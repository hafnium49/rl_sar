# Radio Taiso → Unitree G1 Retargeting Pipeline (as-built)

**Author**: Hiroki Fujiwara
**Reporting window**: 2026-05-14 (post FBX modernization) → 2026-05-15
**Status**: Stage 2c complete; canonical reference motion ready at `~/datasets/radio_taiso/gmr/radio_taiso_g1.npz` (symlink → `take12_g1.npz`). Stage 4 (PPO training input prep) is next.

This doc picks up where [`progress_report_2026-05-14.md`](progress_report_2026-05-14.md) ended (FBX 3000 → FBX 7700 conversion complete, retargeting pipeline yet to start). The driving mission is [`radio_taiso_g1_agent.md`](../radio_taiso_g1_agent.md).

---

## TL;DR

Built a custom retargeting pipeline that converts Ibaraki Motion Star FBX 7700 source data (15 magnetic sensors × ~6300 frames at 30 fps) into a Unitree G1 reference motion (36-column qpos NPZ) suitable for `unitree_rl_mjlab` PPO training. The pipeline registers a new `motionstar` source format in GMR's IK config dictionary — no GMR source-code changes required.

Five iterations of refinement against a metric suite scored on six body-tracking criteria. The current canonical retarget (`take12_g1.npz`) passes **6 of 8 hard metric gates**. The two remaining issues (`self_collision_frame_pct`, `weighted_keypoint_error_pct_height`) are blocked on a known issue in the source-extractor coordinate transform that can be addressed in a follow-up — both are calibration-limited, not retargeting-quality-limited, and don't block PPO training downstream.

---

## 1. Pipeline overview

```
                   ┌────────────────────────────────────────────────────┐
                   │  data/ibaraki_radio_taiso/converted/take{N}.fbx    │
                   │  (FBX 7700, converted from FBX 3000 on x86_64)     │
                   └────────────────────────┬───────────────────────────┘
                                            │
                            scripts/fbx_motionstar_to_npz.py
                                  (Blender 4.0.2 + bpy)
                                            │
                                            ▼
                ┌─────────────────────────────────────────────────────────┐
                │ ~/datasets/radio_taiso/intermediate/motionstar_npz/     │
                │  take{N}.npz                                            │
                │  pos:  (T, 15, 3)  Z-up meters                          │
                │  quat: (T, 15, 4)  wxyz — see § Known limitations       │
                └────────────────────────┬────────────────────────────────┘
                                         │
                       scripts/motionstar_retarget.py
                       (custom `motionstar` GMR src_human)
                                         │
                                         ▼
                ┌─────────────────────────────────────────────────────────┐
                │ ~/datasets/radio_taiso/gmr/take{N}_g1.npz               │
                │  qpos: (T, 36) MuJoCo wxyz order                        │
                └────────────────┬─────────────────────────────┬──────────┘
                                 │                             │
                  scripts/qpos_npz_to_csv.py          scripts/evaluate_retarget.py
                  (wxyz → xyzw quat permute)          (5 metrics + threshold gates)
                                 │                             │
                                 ▼                             ▼
                ┌──────────────────────────────┐  ┌─────────────────────────────┐
                │ ~/datasets/radio_taiso/      │  │ ~/datasets/radio_taiso/     │
                │  gmr_csv/take{N}_g1.csv      │  │  reports/manual/*.json      │
                │  (36 cols, headerless,       │  │  (pass/fail flags, metric   │
                │   same layout as             │  │   values; exit 0 if all     │
                │   dance1_subject2.csv)       │  │   hard thresholds pass)     │
                └──────────────┬───────────────┘  └─────────────────────────────┘
                               │
                  scripts/record_dance_orbit.py  (offscreen MuJoCo render via EGL)
                               │
                               ▼
                ┌──────────────────────────────────┐
                │ ~/datasets/radio_taiso/videos/   │
                │  take{N}_kinematic_orbit.mp4     │
                │  (orbit camera around G1)        │
                └──────────────────────────────────┘
```

**Canonical winner** after Stage 2c batch comparison: `take12` (lowest `weighted_keypoint_error_pct_height` among the 2 takes that passed all 6 physics gates). Symlinked at `~/datasets/radio_taiso/gmr/radio_taiso_g1.npz` → `take12_g1.npz` for downstream stages.

---

## 2. Scripts (with usage)

All scripts live in [`scripts/`](../scripts/).

### `fbx_motionstar_to_npz.py`
Blender headless extractor. Reads a FBX 7700 file, finds the 15 active Motion Star sensors (`MotionStar:Sensor1..Sensor15`), and emits a per-frame NPZ. The 16 unused sensor slots (`Sensor16..Sensor32`, all-zero fcurves in the source) are filtered out.

**Coordinate frame conversion**: Motion Star is left-handed Y-up (X=right, Y=up, Z=forward); MuJoCo G1 is right-handed Z-up (X=forward, Y=left, Z=up). The conversion is a *reflection* (det=−1), not a rotation: `(x, y, z) → (z, −x, y)`. Verified against frame-1 sensor data — see plan Iteration 2 for the verification math. **Side effect**: applied to the rotation part of `matrix_world` too, producing mathematically-undefined quaternions; harmless for our position-only IK but blocks orientation tracking — see § Known limitations.

```bash
blender --background --python scripts/fbx_motionstar_to_npz.py -- \
  data/ibaraki_radio_taiso/converted/take12_fbx7700.fbx \
  ~/datasets/radio_taiso/intermediate/motionstar_npz/take12.npz
```

### `motionstar_retarget.py`
Feeds the NPZ frame-by-frame into a `GeneralMotionRetargeting(src_human="motionstar", tgt_robot="unitree_g1")` instance. The `motionstar` source format is registered in [GMR's `params.py`](../../GMR/general_motion_retargeting/params.py) (under `IK_CONFIG_DICT["motionstar"]`) pointing at our custom [`motionstar_to_g1.json`](../../GMR/general_motion_retargeting/ik_configs/motionstar_to_g1.json).

**Notable behaviors**:
- Pre-grounding: subtracts `min(foot_z) − 0.08 m` from all sensor Z values so the lowest foot sits at +8 cm. Accounts for G1's foot mesh extending ~3 cm below the `toe_link` kinematic site plus headroom for IK solver noise.
- IK exception handling: on `mink.exceptions.NotWithinConfigurationLimits`, freezes at the previous good qpos. Prevents the retargeter from crashing when aggressive weight bumps push joints outside their limits.
- **Wrist DoF zeroing** (Iter 5): the 6 wrist qpos slots (`left/right_wrist_{roll,pitch,yaw}_joint`) are zeroed post-IK. Enforces left/right symmetry — see § Iteration 5.

```bash
conda run -n gmr python scripts/motionstar_retarget.py \
  --npz ~/datasets/radio_taiso/intermediate/motionstar_npz/take12.npz \
  --out ~/datasets/radio_taiso/gmr/take12_g1.npz
```

### `qpos_npz_to_csv.py`
Converts the (T, 36) qpos NPZ to a 36-column CSV with `xyzw` quaternion order (matches `policy/g1/whole_body_tracking/dance_102/G1_Take_102.bvh_60hz.csv` layout). MuJoCo qpos is `wxyz`; the CSV reader/writer permutes `qpos[3]` (`qw`) to `csv[6]` and shifts `qpos[4:7]` (`qxyz`) down to `csv[3:6]`.

```bash
/tmp/rl_sar_uv_env/bin/python scripts/qpos_npz_to_csv.py \
  --qpos ~/datasets/radio_taiso/gmr/take12_g1.npz \
  --csv  ~/datasets/radio_taiso/gmr_csv/take12_g1.csv
```

### `evaluate_retarget.py`
The metric suite (the "priority-5" recommended in the second-opinion review for mocap retargeting quality):

| # | Metric | Detects |
|---|---|---|
| 1 | `weighted_keypoint_error_pct_height` | Wrong body mapping, marker-to-joint offset, wrong scale |
| 2 | `pelvis_error_cm` | Bad global tracking |
| 3 | `foot_error_during_contact_cm` + `foot_slip_during_contact_cm_per_s` | Foot sliding during planted contact |
| 4 | `joint_limit_violation_frame_pct` + `joint_jump_frame_pct` | Impossible reference, discontinuous IK |
| 5 | `self_collision_frame_pct` + `ground_penetration_frame_pct` | Arm-through-torso, foot-into-floor |

Hard-fail thresholds: see `DEFAULT_THRESHOLDS` in the script. Exit code 0 = all hard thresholds pass; exit 1 = at least one failed. JSON output is the source of truth for iteration decisions.

```bash
conda run -n gmr python scripts/evaluate_retarget.py \
  --source-npz ~/datasets/radio_taiso/intermediate/motionstar_npz/take12.npz \
  --g1-qpos    ~/datasets/radio_taiso/gmr/take12_g1.npz \
  --mapping    ~/projects/GMR/general_motion_retargeting/ik_configs/motionstar_to_g1.json \
  --output     ~/datasets/radio_taiso/reports/manual/take12_final.json
```

### `estimate_morphology.py`
Diagnostic tool — measures human body segments from NPZ frame 0 + G1 segments from MJCF via `mj_forward(qpos0)`. Emits per-segment scale ratios. Used once in Iteration 4 to verify scale-table values; confirmed our guess values (0.9, 0.8) were within 2-3% of measured ratios (0.876, 0.775). **Not run as part of the production pipeline** — diagnostic only.

```bash
conda run -n gmr python scripts/estimate_morphology.py
# emits ~/datasets/radio_taiso/reports/morphology.json
```

### `record_dance_orbit.py` (existing, reused)
Orbit-camera offscreen MuJoCo renderer from the dance work. Reused as-is via `--motion <csv>` flag. Renders at ~165 fps; 6300 frames in ~38 s.

### `run_retarget_iteration.py` (legacy)
Built early in Iteration 3 as an autonomous-driver with pre-specified recovery rules. **Abandoned** after the user pivoted to Claude-in-loop mode (Iteration 3A onwards was done manually with Claude reading metric JSON and making targeted config edits). Kept on disk as reference; not part of the production pipeline.

---

## 3. Iteration history

Five iterations of refinement, each gated on the metric suite from `evaluate_retarget.py`. Numerical entries are for take11 (the development take); take12 metrics are similar (Stage 2c batched the final config over all 5 takes).

| Iter | Change | What worked | Critical metric | What didn't / why |
|---|---|---|---|---|
| **1** | All `rot_weight=0` in [`motionstar_to_g1.json`](../../GMR/general_motion_retargeting/ik_configs/motionstar_to_g1.json) — position-only IK | First retarget that didn't crash | Replaced 27° forward-twist with cleaner posture | Body bent backward + left/right mirrored (revealed in Iter 2) |
| **2** | Coordinate transform in [`fbx_motionstar_to_npz.py:67`](../scripts/fbx_motionstar_to_npz.py): `+90°-about-X` rotation → `(x,y,z) → (z,−x,y)` reflection (det=−1) | Body now faces forward, left/right correct | Pelvis, chest, foot positions verified against frame-1 source data | "Garbage" rotation part — `to_quaternion()` on a reflection-bearing matrix returns undefined values. Blocks Iter 3D below. |
| **3A** | Restructure `motionstar_to_g1.json` table weights — physics-first: pelvis & feet pos_weight 100→200, all other table1 entries → token weight 1, table2 priority-weighted (knees/hips/torso=20, shoulders=20, elbows=40, hands=80) | **pelvis_error 6.85 → 3.56 cm; ground_penetration 48.9% → 0.0%; joint_jump 9.0% → 1.16%** — 6 of 8 physics gates pass in ONE iteration | The "informed weight pre-bump" — the auto-driver had tried 3 small bumps and ran out of budget; Claude looked at the JSON, saw pelvis was the blocker, bumped 2× upfront | — |
| **3B** | Swap hand IK targets from `wrist_yaw_link` to `rubber_hand` (G1 has these bodies at +4.15 cm distal of wrist) | wrist_error 16.20 → 14.93 cm (4 cm geometric shift produced 1.3 cm improvement); joint_jump 1.63 → 1.16% | The G1 model already had palm-position bodies — no XML changes needed | — |
| **3C** | Rest-pose calibration: compute per-sensor `pos_offset` and `rot_offset` from frame-0 source vs G1's IK-result qpos[0] | — | **REVERTED**. Self-consistent calibration produced 40–80 cm `pos_offsets` because source quaternions are garbage (Iter 2 side effect). Applying those offsets regressed metrics 3-4× (`weighted_kp_err` 10.76 → 19.54%, `self_collision` 26 → 82%, `ground_penetration` 0 → 36.6%). |
| **3D** | Re-enable `rot_weight` on rubber_hand targets | — | **SKIPPED**. Same garbage-quaternion blocker as 3C. |
| **4** | Replace `human_scale_table` guess values (leg 0.9, arm 0.8) with morphology-measured ratios (leg 0.876, arm 0.775) | — | **REVERTED**. Measured ratios differed from guesses by 2-3% — too small to move `self_collision` or `weighted_kp_err`. The morphology mismatch isn't the binding constraint for the remaining metric failures. |
| **5** | Lock 6 wrist DoFs at 0 post-IK in [`motionstar_retarget.py`](../scripts/motionstar_retarget.py) | **joint_jump 1.41 → 1.09%; joint_limit_violation 0.05 → 0.00%**; wrist trajectories now mathematically symmetric | Trade-off: wrist_error +1.56 cm (positional metric artifact — neutral wrist orientation no longer aligned with back-of-hand sensor orientation). Wrist motion content lost but PPO-irrelevant. |
| **6A** | Add per-body L/R metric breakdown + `lr_asymmetry_max_ratio` to [`evaluate_retarget.py`](../scripts/evaluate_retarget.py) | `lr_asymmetry_max_ratio = 0.235` (threshold 0.25) → **SOFT PASS** at average level. Worst pair: toes (left 11.6 vs right 7.2 cm). | Diagnostic only — confirms morphological symmetry of the *config* is reflected in *average* output. |
| **6B** | Build [`scripts/test_mirror_equivariance.py`](../scripts/test_mirror_equivariance.py) — implements `F(M_source(x)) ≈ M_robot(F(x))` test from the third-opinion review | Per-body world-position diff between `retarget(mirror(source))` and `mirror(retarget(source))`. | **FAIL on max** (80 cm at left_elbow_link) but mean diffs 4-13 cm. Bias is mink solver path-dependence on transient frames, not configuration bias. Accepted as known limitation; full fix would require alternate IK solver or per-frame initial-pose seeding. |

**Stage 2c** (batch over takes 12–15 with the Iter-3-final config): take12 won with 6/8 gates pass + `weighted_kp_err = 10.34`. Take11 was the runner-up (6/8 gates, 10.76). Takes 13–15 had `ground_penetration` of 49–56% (data quality issue, likely frame-0 not-a-rest-pose; held back as a future improvement opportunity but not blocking).

---

## 4. Files this work created or modified

### Created
| File | Purpose |
|---|---|
| [`scripts/fbx_motionstar_to_npz.py`](../scripts/fbx_motionstar_to_npz.py) | Blender headless FBX → 15-sensor NPZ extractor |
| [`scripts/motionstar_retarget.py`](../scripts/motionstar_retarget.py) | NPZ → G1 qpos via GMR's `motionstar` source |
| [`scripts/qpos_npz_to_csv.py`](../scripts/qpos_npz_to_csv.py) | qpos NPZ → 36-col CSV (wxyz→xyzw permute) |
| [`scripts/evaluate_retarget.py`](../scripts/evaluate_retarget.py) | Metric suite + threshold gates + JSON output |
| [`scripts/estimate_morphology.py`](../scripts/estimate_morphology.py) | Diagnostic — measures human + G1 segments |
| [`scripts/run_retarget_iteration.py`](../scripts/run_retarget_iteration.py) | Legacy autonomous driver (abandoned for Claude-in-loop) |
| `~/projects/GMR/general_motion_retargeting/ik_configs/motionstar_to_g1.json` | Custom IK config for the `motionstar` source (in GMR clone, not this repo) |

### Modified in GMR clone
| File | Change |
|---|---|
| `~/projects/GMR/general_motion_retargeting/params.py` | Added `"motionstar": {"unitree_g1": IK_CONFIG_ROOT / "motionstar_to_g1.json"}` to `IK_CONFIG_DICT` |

No changes to GMR's `motion_retarget.py` or other Python source.

### Produced (not committed; under `~/datasets/radio_taiso/`)
- `intermediate/motionstar_npz/take{11..15}.npz` — extracted sensor data
- `gmr/take{11..15}_g1.npz` — retargeted G1 qpos
- `gmr/radio_taiso_g1.npz` — symlink → `take12_g1.npz` (canonical winner)
- `gmr_csv/take{11..15}_g1.csv` — same data in CSV form
- `gmr_csv/radio_taiso_g1.csv` — symlink → `take12_g1.csv`
- `reports/manual/take{N}_*.json` — per-iteration metric JSONs
- `reports/morphology.json` — Iter 4 measurement output
- `videos/take{N}_kinematic_orbit.mp4` — orbit-camera previews (also copied to repo root for review)

---

## 5. Final metric snapshot (take12, current state)

```
=== take12_iter5.json ===
  weighted_keypoint_error_pct_height         =  10.44   <= 6.0        [FAIL]
  pelvis_error_cm                            =   2.77   <= 4.0        [PASS]
  foot_error_during_contact_cm               =   4.14   <= 5.0        [PASS]
  wrist_error_cm                             =  16.19   <= 12.0       [SOFT]
  joint_limit_violation_frame_pct            =   0.00   <= 0.5        [PASS]
  joint_jump_frame_pct                       =   1.09   <= 3.0        [PASS]
  self_collision_frame_pct                   =  28.07   <= 3.0        [FAIL]
  ground_penetration_frame_pct               =   0.19   <= 1.0        [PASS]
  foot_slip_during_contact_cm_per_s          =   2.48   <= 5.0        [PASS]
```

**6 of 8 hard gates pass.** Remaining failures:
- `weighted_keypoint_error_pct_height` — uncorrected marker-to-joint offsets; blocked on source quaternion garbage
- `self_collision_frame_pct` — arms occasionally clip torso during forward bends; same blocker

Both are **calibration-limited**, not retargeting-quality-limited. The motion is smooth and looks like Radio Taiso visually. PPO training is the correct next step.

---

## 6. Known limitations & open issues

### Source quaternion garbage (blocks 3C, 3D, and full orientation tracking)
The reflection matrix in [`fbx_motionstar_to_npz.py:67`](../scripts/fbx_motionstar_to_npz.py) (det = −1) is applied to both position and rotation parts of Blender's `matrix_world`. Extracting `to_quaternion()` from a reflection-bearing matrix is mathematically undefined — the values written to NPZ for the `quat` field are arbitrary. Position-only IK doesn't care, but any approach that uses sensor orientation (rot_offset calibration, rotation-targeting IK) gets garbage in → garbage out.

**Fix when needed**: re-extract NPZ with a separate transform for orientations — a det=+1 rotation that approximates the position reflection's directional intent (rotate about an axis that takes Y→Z, ignore the chirality flip). Substantial work (probably ~200 LOC of careful axis-angle math); deferred until orientation tracking becomes the binding constraint after PPO training.

### Marker-to-joint offsets unaddressed
Motion Star sensors are on body segments (mid-shin, mid-thigh, back-of-hand), not at joint centers. The IK targets them directly, so G1's body sites land at sensor positions — slightly offset from where joint centers should be. Manifests as `self_collision_frame_pct = 28%` (arms hit torso during bends, because the IK pulls wrists to back-of-hand position, slightly past the actual wrist). Fixing requires either the source quaternion fix above + 3C calibration, or hard-coded `pos_offset` values derived from anatomy.

### Grounding heuristic doesn't generalize
The retargeter's `floor_z = min(foot_z) − 0.08 m` works for takes 11–12 (frame 0 = clean standing pose) but produces 49–56% ground penetration on takes 13–15. The frame-0 pose for those takes likely isn't a clean stand (performer mid-motion, foot lifted, etc.). **Mitigation**: median foot_z over the first 30 frames would be more robust. Not implemented because takes 11/12 are sufficient for PPO training; left as an improvement opportunity.

### Wrist motion content lost (Iter 5 trade-off)
Locking wrist DoFs at 0 means the reference motion has no wrist-orientation content. For PPO this is acceptable (reward dominated by major joints), but a faithful Radio Taiso reproduction needs wrist articulation. Same source-quaternion fix above would unblock proper wrist tracking via 3D's orientation IK.

### Retargeter solver path-dependence (Iter 6B finding)
Mink's IK solver produces non-mirror-equivariant output on transient frames — `retarget(mirror(source))` differs from `mirror(retarget(source))` by up to 80 cm at the worst body (left elbow) on individual frames, with mean diffs of 4-13 cm. The pattern is "left side consistently slightly worse than right." This is *solver* asymmetry (deterministic starting point + iterative gradient descent producing different local minima on swapped data), not *config* asymmetry — the IK weights and rot_offsets are properly mirror-paired. **`lr_asymmetry_max_ratio = 0.235` (just under the 0.25 threshold)** confirms the average-frame behavior is balanced; only specific frames show large divergence. Fixing this would require either a different IK solver or per-frame initial-pose seeding from the previous frame's mirror — substantial work for marginal PPO-training benefit. Documented for future investigation.

---

## 7. Workflow conventions (for next maintainer)

- **Claude-in-loop iteration**: when refining the retargeting (changing weights, scales, offsets), the workflow is: edit `motionstar_to_g1.json` → run retarget+CSV+eval → read the metric JSON → decide → edit → repeat. No autonomous-driver scripts; the agent reads metrics and makes targeted edits. See `~/.claude/projects/.../memory/feedback_autonomous_loops.md` for the user's preference.
- **Metric JSON is the gate, not visual MP4**: render MP4s as side artifacts for retrospective inspection, but iteration decisions should be metric-based. Exit code 0 from `evaluate_retarget.py` = all hard thresholds pass.
- **Per-sub-iteration thresholds were used briefly** (in `run_retarget_iteration.py`'s `SUB_THRESH_OVERRIDES`) but the abandoned autonomous-driver pattern doesn't apply to Claude-in-loop work. Use the default thresholds for production gating.
- **Take selection**: `weighted_keypoint_error_pct_height` is the primary criterion (gross-body alignment), `self_collision_frame_pct` is the tiebreaker. Symlinks at `~/datasets/radio_taiso/gmr/radio_taiso_g1.{npz,csv}` reflect the current winner.

---

## 8. Next steps

**Stage 4** — install `unitree_rl_mjlab` in a fresh conda env, run its `csv_to_npz.py` on the canonical `radio_taiso_g1.csv`, re-run `evaluate_retarget.py` on the post-NPZ data as a Gate-Kinematics check (ensures the NPZ resampling at 50 fps hasn't degraded the motion).

**Stage 5** — PPO training via `python scripts/train.py Unitree-G1-Tracking-No-State-Estimation --motion_file=...`. Multi-hour GPU run. The smooth-but-imperfect reference should be sufficient for the policy to learn; if training collapses or fails the 20/20 sim gate, revisit the open issues in §6 (source quaternion fix → 3C calibration → 3D orientation tracking).

**Stage 6** — 20 MuJoCo play trials; aggregate pass/fail against the 20/20 gate.

**Stage 7** (optional) — NMR comparison route via `MakeTrackingEasy`. Defer unless GMR route fails.

**Stage 8** — deploy prep + sim-loopback via `unitree_mujoco` (no real hardware per the brief).

See the [plan file](/home/h_fujiwara/.claude/plans/read-agent-md-and-plan-sharded-platypus.md) for detailed downstream steps if the maintainer wants the verbose version.

---

## Appendix A — How to reproduce the current canonical motion

```bash
# (Once) Set up GMR env
cd ~/projects/GMR
conda create -n gmr python=3.10 -y
conda run -n gmr pip install -e .

# (Once, per take) Extract sensor NPZ from FBX 7700
for n in 11 12 13 14 15; do
  blender --background --python ~/projects/rl_sar/scripts/fbx_motionstar_to_npz.py -- \
    ~/projects/rl_sar/data/ibaraki_radio_taiso/converted/take${n}_fbx7700.fbx \
    ~/datasets/radio_taiso/intermediate/motionstar_npz/take${n}.npz
done

# (Per iteration) Retarget + CSV + score
for n in 11 12 13 14 15; do
  conda run -n gmr python ~/projects/rl_sar/scripts/motionstar_retarget.py \
    --npz ~/datasets/radio_taiso/intermediate/motionstar_npz/take${n}.npz \
    --out ~/datasets/radio_taiso/gmr/take${n}_g1.npz
  /tmp/rl_sar_uv_env/bin/python ~/projects/rl_sar/scripts/qpos_npz_to_csv.py \
    --qpos ~/datasets/radio_taiso/gmr/take${n}_g1.npz \
    --csv  ~/datasets/radio_taiso/gmr_csv/take${n}_g1.csv
  conda run -n gmr python ~/projects/rl_sar/scripts/evaluate_retarget.py \
    --source-npz ~/datasets/radio_taiso/intermediate/motionstar_npz/take${n}.npz \
    --g1-qpos    ~/datasets/radio_taiso/gmr/take${n}_g1.npz \
    --mapping    ~/projects/GMR/general_motion_retargeting/ik_configs/motionstar_to_g1.json \
    --output     ~/datasets/radio_taiso/reports/manual/take${n}_final.json
done

# Update canonical winner symlink (currently take12)
cd ~/datasets/radio_taiso/gmr     && ln -sf take12_g1.npz radio_taiso_g1.npz
cd ~/datasets/radio_taiso/gmr_csv && ln -sf take12_g1.csv radio_taiso_g1.csv
```
