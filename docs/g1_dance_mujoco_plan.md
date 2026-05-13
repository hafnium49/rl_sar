# Plan: Launch G1 dance_102 in MuJoCo via rl_sar

## Context

[agent.md](/home/h_fujiwara/projects/rl_sar/agent.md) is a mission brief for running a **pretrained Unitree G1 whole-body-tracking dance policy in MuJoCo simulation** using the [rl_sar](/home/h_fujiwara/projects/rl_sar) repo — simulation-only, no training, no real hardware. Target policy (per user): **`policy/g1/whole_body_tracking/dance_102`**. Intended outcome: G1 stands up in MuJoCo, then plays the `G1_Take_102.bvh_60hz.csv` reference motion under the pretrained 154-obs policy, ideally for ≥60 s without falling. The session must end with a structured report at `rl_sar_g1_dance_run_report.md`.

The current working tree is clean and unbuilt:
- `library/mujoco/` — absent
- `library/inference_runtime/` — absent
- `src/rl_sar_zoo/` — absent (this is where the G1 MuJoCo scene `g1_description/mjcf/scene_29dof.xml` lives)
- `cmake_build/` — absent

Per user direction, `./build.sh -mj` will fetch these automatically via its internal calls to [scripts/download_inference_runtime.sh](/home/h_fujiwara/projects/rl_sar/scripts/download_inference_runtime.sh), [scripts/download_robot_descriptions.sh](/home/h_fujiwara/projects/rl_sar/scripts/download_robot_descriptions.sh), and [scripts/download_mujoco.sh](/home/h_fujiwara/projects/rl_sar/scripts/download_mujoco.sh).

## What's already verified

- The policy [policy/g1/whole_body_tracking/dance_102/](/home/h_fujiwara/projects/rl_sar/policy/g1/whole_body_tracking/dance_102) is self-contained: `policy.pt` (978K), `G1_Take_102.bvh_60hz.csv` (582K), `config.yaml`. No additional download needed for the policy itself.
- The G1 FSM at [src/rl_sar/fsm_robot/fsm_g1.hpp](/home/h_fujiwara/projects/rl_sar/src/rl_sar/fsm_robot/fsm_g1.hpp) hard-codes the skill→key mapping. `RLFSMStateRLWholeBodyTrackingDance102` is entered from `RLFSMStateGetUp` on **Num3 / RB_DPadLeft**. It sets `rl.config_name = "whole_body_tracking/dance_102"` (fsm_g1.hpp:290), which `InitRL` then resolves under `POLICY_DIR/g1/`.
- The MuJoCo binary at [src/rl_sar/src/rl_sim_mujoco.cpp:64](/home/h_fujiwara/projects/rl_sar/src/rl_sar/src/rl_sim_mujoco.cpp) resolves `<ROBOT> <SCENE>` to `<repo>/src/rl_sar_zoo/<ROBOT>_description/mjcf/<SCENE>.xml` — i.e. for the launch `g1 scene_29dof` it loads `src/rl_sar_zoo/g1_description/mjcf/scene_29dof.xml`.
- Motion CSV is loaded by [src/rl_sar/library/core/motion_loader/motion_loader.cpp](/home/h_fujiwara/projects/rl_sar/src/rl_sar/library/core/motion_loader/motion_loader.cpp) from `POLICY_DIR/g1/whole_body_tracking/dance_102/G1_Take_102.bvh_60hz.csv`. Missing file → `std::runtime_error("Failed to open motion file: ...")`.

## Execution steps

### 1 — Pre-flight: confirm host deps and display

```bash
# Display server reachable (MuJoCo viewer needs GLX)
echo "$DISPLAY"
glxinfo | grep "OpenGL renderer" || echo "WARN: glxinfo missing — viewer may fail"

# Compile deps (from README §Dependency)
dpkg -l | grep -E '^ii\s+(cmake|g\+\+|libyaml-cpp-dev|libeigen3-dev|libboost-all-dev|libspdlog-dev|libfmt-dev|libtbb-dev|liblcm-dev)\s' | wc -l
```

If any of the eight apt packages are missing, install them per [README.md:78](/home/h_fujiwara/projects/rl_sar/README.md). Record the result in the report.

### 2 — Build with auto-fetched assets

```bash
cd /home/h_fujiwara/projects/rl_sar
./build.sh -mj 2>&1 | tee build_mujoco.log
```

This triggers, in order: `setup_inference_runtime` (LibTorch + ONNX Runtime → `library/inference_runtime/`), `setup_robot_descriptions` (clones `fan-ziqi/rl_sar_zoo` to `src/rl_sar_zoo/`), `setup_mujoco` (MuJoCo 3.2.7 prebuilt → `library/mujoco/`), then `run_mujoco_build` (CMake into `cmake_build/`). Expect first-time downloads totaling several hundred MB.

**Success criteria for this step:**
- `cmake_build/bin/rl_sim_mujoco` exists and is executable
- `src/rl_sar_zoo/g1_description/mjcf/scene_29dof.xml` exists
- `library/mujoco/` and `library/inference_runtime/libtorch/` (or `onnxruntime/`) populated
- Build log ends without an unresolved compiler/linker error

If build fails, capture the first compiler/linker error from `build_mujoco.log` into the report's "Build result" section and stop — do **not** patch sources.

### 3 — Launch the MuJoCo simulation

```bash
cd /home/h_fujiwara/projects/rl_sar
./cmake_build/bin/rl_sim_mujoco g1 scene_29dof 2>&1 | tee run.log
```

A MuJoCo viewer window should open with the G1 in its initial program pose. The FSM starts in `RLFSMStatePassive` ([fsm_g1.hpp:15](/home/h_fujiwara/projects/rl_sar/src/rl_sar/fsm_robot/fsm_g1.hpp)) — motors held at `kp=0, kd=8` (intentional damping, not a bug).

If the viewer does not open, capture `echo $DISPLAY`, `ldd ./cmake_build/bin/rl_sim_mujoco | grep "not found"`, and any GL/GLX error from `run.log` into the report, then stop.

### 4 — In-sim control sequence

With the MuJoCo window focused:

| Step | Key | Expected effect |
|---|---|---|
| 1 | `Num0` | FSM: Passive → GetUp. Robot interpolates over ~2 s to `default_dof_pos` (slight hip pitch, knees bent, arms at sides). |
| 2 | *(wait ~3 s)* | Robot stabilizes standing under `fixed_kp/fixed_kd` from [policy/g1/base.yaml](/home/h_fujiwara/projects/rl_sar/policy/g1/base.yaml). |
| 3 | `Num3` | FSM: GetUp → RLWholeBodyTrackingDance102. Loads `policy.pt` + `G1_Take_102.bvh_60hz.csv` (csv missing here would crash — recovery path below). Terminal prints `RL Controller [whole_body_tracking/dance_102] x:0 y:0 yaw:0`. |
| 4 | *(observe)* | Robot performs the dance, reference motion auto-loops once duration is reached and reverts to locomotion. |

Safety / recovery keys: `R` resets the MuJoCo world; `Num9` returns to initial pose; `P` (or `LB+X` gamepad) drops back to Passive.

### 5 — Capture observation data

While the dance is running, capture for the report:

- Wall-clock duration the robot stays upright before any fall or termination
- Any "Failed to open motion file", "InitRL() failed", or torque-protect warnings (`TorqueProtect` from [rl_sdk.hpp:250](/home/h_fujiwara/projects/rl_sar/src/rl_sar/library/core/rl_sdk/rl_sdk.hpp))
- MuJoCo "elastic" / contact warnings if any
- Approximate real-time factor from the viewer's status bar
- Screenshots: standing pose after Num0, mid-dance frame, and final state. Save under `/tmp/rl_sar_dance_screens/` and reference paths in the report.

### 6 — Produce the report

Write `/home/h_fujiwara/projects/rl_sar/rl_sar_g1_dance_run_report.md` with the nine sections from [agent.md §Deliverables](/home/h_fujiwara/projects/rl_sar/agent.md):

1. **Environment** — Linux kernel (`uname -r`), NVIDIA driver if present (`nvidia-smi --query-gpu=name,driver_version --format=csv`), Display server
2. **Repository state** — `git rev-parse HEAD`, `git status -s`, `git submodule status`
3. **Build result** — exact `./build.sh -mj` command, success/failure, `cmake_build/bin/rl_sim_mujoco` size, first error if failed
4. **Policy discovery** — `find policy/g1 -type f`, selected = `policy/g1/whole_body_tracking/dance_102/policy.pt`, format = LibTorch JIT (.pt), config path
5. **Run command** — `./cmake_build/bin/rl_sim_mujoco g1 scene_29dof`
6. **Skill activation** — key sequence `Num0` → wait → `Num3`; mapping discovered in `fsm_g1.hpp:280–370`
7. **Observed behavior** — default-pose result, dance-start result, upright duration, failure modes
8. **Artifacts** — paths to `build_mujoco.log`, `run.log`, screenshots dir
9. **Next actions** — concrete next steps based on what actually happened (e.g., "robot fell at t≈8 s during torso rotation; inspect waist kp/kd in config.yaml" — only if true)

### 7 — Optional follow-up (stretch goal)

If dance_102 completes ≥60 s upright, document it and (per user's earlier preference) **do not** auto-chain into gangnam_style — the user asked for dance_102 first only. Note "gangnam_style not yet tested" as a next action.

## Critical files (read-only references for execution)

- [build.sh](/home/h_fujiwara/projects/rl_sar/build.sh) — orchestrator; `-mj` path is `setup_inference_runtime` → `setup_robot_descriptions` → `setup_mujoco` → `run_mujoco_build`
- [src/rl_sar/CMakeLists.txt:622-713](/home/h_fujiwara/projects/rl_sar/src/rl_sar/CMakeLists.txt) — `USE_MUJOCO` branch, defines `rl_sim_mujoco` executable
- [src/rl_sar/src/rl_sim_mujoco.cpp:64](/home/h_fujiwara/projects/rl_sar/src/rl_sar/src/rl_sim_mujoco.cpp) — scene XML path template
- [src/rl_sar/fsm_robot/fsm_g1.hpp:280-370](/home/h_fujiwara/projects/rl_sar/src/rl_sar/fsm_robot/fsm_g1.hpp) — `RLFSMStateRLWholeBodyTrackingDance102::Enter()` (line 290 sets `config_name`)
- [policy/g1/whole_body_tracking/dance_102/config.yaml](/home/h_fujiwara/projects/rl_sar/policy/g1/whole_body_tracking/dance_102/config.yaml) — 154 obs, custom joint_mapping, `motion_file: G1_Take_102.bvh_60hz.csv`
- [policy/g1/base.yaml](/home/h_fujiwara/projects/rl_sar/policy/g1/base.yaml) — 29-DoF physical joint order, `fixed_kp/fixed_kd` used in Passive/GetUp

## Files this plan will create (none modify existing code)

- `/home/h_fujiwara/projects/rl_sar/build_mujoco.log` (build output capture)
- `/home/h_fujiwara/projects/rl_sar/run.log` (sim runtime capture)
- `/tmp/rl_sar_dance_screens/` (screenshots)
- `/home/h_fujiwara/projects/rl_sar/rl_sar_g1_dance_run_report.md` (final deliverable)
- `library/mujoco/`, `library/inference_runtime/`, `src/rl_sar_zoo/`, `cmake_build/` (created by build script, gitignored)

**No existing source files will be edited.** No changes to `policy/`, `src/rl_sar/`, or any committed file.

## Verification

End-to-end check after executing the plan:

```bash
# 1. Binary exists
test -x /home/h_fujiwara/projects/rl_sar/cmake_build/bin/rl_sim_mujoco && echo "binary OK"

# 2. Scene XML present
test -f /home/h_fujiwara/projects/rl_sar/src/rl_sar_zoo/g1_description/mjcf/scene_29dof.xml && echo "scene OK"

# 3. Motion file readable (the runtime check that determines success/failure of skill load)
test -r /home/h_fujiwara/projects/rl_sar/policy/g1/whole_body_tracking/dance_102/G1_Take_102.bvh_60hz.csv && echo "motion CSV OK"

# 4. Report exists with all nine sections
grep -c "^## " /home/h_fujiwara/projects/rl_sar/rl_sar_g1_dance_run_report.md
# expect: ≥ 9
```

The interactive verification is the human-loop observation in step 5: did the robot stand on Num0, did Num3 start the dance, and how long did it stay upright. Those answers are the load-bearing content of the final report.

## Out of scope (explicit non-goals per agent.md)

- Training any new policy
- Real-hardware deployment (no `rl_real_g1` execution)
- Custom dance / kung-fu retargeting pipeline
- Patching `src/` to "fix" failures — if a failure occurs, document it and stop
