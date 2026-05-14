# agent.md — Claude Code Agent: Run Pretrained Unitree G1 Dance in `rl_sar`

## Mission

You are Claude Code operating as a robotics deployment assistant. Your mission is to set up and run a **pretrained Unitree G1 long-horizon dance controller** in **MuJoCo simulation** using the open-source repository:

```text
https://github.com/fan-ziqi/rl_sar
```

The target is **simulation-only** execution of a pretrained RL whole-body tracking policy, such as:

```text
policy/g1/whole_body_tracking/gangnam_style
policy/g1/whole_body_tracking/dance_102
```

The expected architecture is:

```text
reference dance motion
  +
pretrained RL whole-body motion-tracking policy
  +
deterministic deployment wrapper
  +
MuJoCo simulation
```

The goal is **not** to train a new policy. The first objective is to launch the G1 digital twin, bring it to a safe standing pose, and trigger a pretrained dance skill in simulation.

---

## Hard Constraints

1. **Simulation only.** Do not attempt real Unitree G1 deployment.
2. **Do not train a new policy.** Use pretrained policies already present in the repository or documented by the repository.
3. **Do not invent missing scripts.** The expected workflow is C++/CMake/MuJoCo based, not a generic Python `deploy_mujoco.py` flow.
4. **Do not rewrite the repository architecture unless necessary.** First inspect the existing README, build scripts, policy folders, and executable names.
5. **Do not assume skill-key mappings.** Inspect the configuration and source code to identify how `gangnam_style` or `dance_102` is selected.
6. **Stop if there is any sign of real-hardware command transmission.** The intended target is local MuJoCo simulation only.

---

## Technical Background

`rl_sar` is a deployment-oriented framework for simulation verification and physical deployment of robot reinforcement-learning controllers. For this task, use only the simulation path.

The relevant robot and task are:

```text
Robot:      Unitree G1
Model:      29-DoF G1 scene
Simulator:  MuJoCo
Policies:   G1 whole-body tracking policies
Examples:   gangnam_style, dance_102
```

The controller is not a pure rule-based trajectory replay. It is a pretrained RL tracking policy wrapped by deterministic logic for:

- initial pose handling,
- interpolation to default pose,
- finite-state-machine transitions,
- motor enable/disable,
- reset,
- passive mode,
- skill switching,
- safety-oriented sequencing.

---

## Expected Workflow Summary

```text
1. Clone `rl_sar` recursively.
2. Install Ubuntu C++/CMake dependencies.
3. Build MuJoCo support with `./build.sh -mj`.
4. Confirm G1 dance policy folders exist.
5. Launch G1 MuJoCo simulation.
6. Bring robot to default standing pose.
7. Identify and trigger the dance skill.
8. Record commands, logs, observations, and any failure modes.
```

---

## Environment Assumptions

Prefer Ubuntu 22.04 or a Linux environment compatible with MuJoCo visualization.

If running under WSL2, verify GUI support before starting. If the MuJoCo viewer cannot open, document the error and suggest running on native Ubuntu or via Docker/X11 forwarding.

Do not start with a Python Conda environment unless the repository explicitly requires it. The main current path is C++/CMake-first.

---

## Step 1 — Clone Repository with Submodules

Run:

```bash
git clone --recursive --depth 1 https://github.com/fan-ziqi/rl_sar.git
cd rl_sar
```

If the repository is already cloned, update it safely:

```bash
git pull
git submodule update --init --recursive --recommend-shallow --progress
```

Verify structure:

```bash
pwd
ls
ls policy || true
ls policy/g1 || true
```

Record the current commit:

```bash
git rev-parse HEAD
```

---

## Step 2 — Inspect Documentation Before Building

Before installing or building, inspect these files if present:

```bash
ls README* docs 2>/dev/null || true
sed -n '1,220p' README.md
find . -maxdepth 3 -iname '*readme*' -o -iname '*.md' | sort
```

Look specifically for:

- MuJoCo build instructions,
- dependency list,
- `build.sh` options,
- G1 policy names,
- G1 scene names,
- keyboard/gamepad controls,
- FSM or skill-switching docs,
- policy loading conventions.

Do not proceed blindly if the repository structure has changed. Adapt only after reading the local repo.

---

## Step 3 — Install Ubuntu Dependencies

Install likely dependencies:

```bash
sudo apt update

sudo apt install -y \
  cmake \
  g++ \
  build-essential \
  libyaml-cpp-dev \
  libeigen3-dev \
  libboost-all-dev \
  libspdlog-dev \
  libfmt-dev \
  libtbb-dev \
  liblcm-dev
```

If build errors report missing packages, install only what is needed and record the exact error and package fix.

---

## Step 4 — Build MuJoCo Support

The expected command is:

```bash
./build.sh -mj
```

If a clean rebuild is needed:

```bash
./build.sh -c
./build.sh -mj
```

After build, inspect generated binaries:

```bash
find cmake_build -maxdepth 4 -type f -executable | sort
find . -path '*bin*' -type f -executable | sort | head -100
```

Expected MuJoCo binary pattern:

```text
./cmake_build/bin/rl_sim_mujoco
```

If this binary is absent, inspect `build.sh`, `CMakeLists.txt`, and README for the new output path.

---

## Step 5 — Confirm G1 Dance Policies

Find available G1 policy folders:

```bash
find policy/g1 -maxdepth 5 -type d | sort
```

Expected folders may include:

```text
policy/g1/whole_body_tracking/gangnam_style
policy/g1/whole_body_tracking/dance_102
```

Inspect policy files:

```bash
find policy/g1 -type f | sort | sed -n '1,240p'
find policy/g1 -name 'config.yaml' -o -name 'base.yaml' -o -name '*.pt' -o -name '*.onnx'
```

Inspect relevant configs:

```bash
sed -n '1,240p' policy/g1/base.yaml 2>/dev/null || true
sed -n '1,240p' policy/g1/whole_body_tracking/gangnam_style/config.yaml 2>/dev/null || true
sed -n '1,240p' policy/g1/whole_body_tracking/dance_102/config.yaml 2>/dev/null || true
```

Record:

- policy path,
- policy file name,
- model format (`.pt` or `.onnx`),
- action dimension,
- observation dimension if visible,
- control frequency,
- PD gains if visible,
- joint order if visible.

---

## Step 6 — Run G1 MuJoCo Simulation

Expected command:

```bash
./cmake_build/bin/rl_sim_mujoco g1 scene_29dof
```

The documented command pattern is expected to be:

```bash
./cmake_build/bin/rl_sim_mujoco <ROBOT> <SCENE>
```

If `scene_29dof` fails, inspect available scene names:

```bash
find . -iname '*scene*' -o -iname '*.xml' | sort
find . -iname '*g1*' | sort | sed -n '1,240p'
```

Then retry with the correct local scene name.

---

## Step 7 — Bring Robot to Default Pose

After MuJoCo viewer opens, use the documented keyboard/gamepad state-machine controls.

Expected common controls:

```text
Num0  -> interpolate from initial program pose to default_dof_pos
Num9  -> interpolate back to initial program pose
R     -> reset simulation if the robot falls
M     -> motor enable
K     -> motor disable
P     -> passive motor mode
Space -> zero movement commands
```

First bring the robot to default standing pose:

```text
1. Launch MuJoCo.
2. If necessary, press R to reset.
3. Press Num0 to interpolate to default standing pose.
4. Observe whether G1 stabilizes.
```

Record whether the robot stands without falling before triggering dance.

---

## Step 8 — Identify Skill Switching for Dance

Expected skill switching keys may be:

```text
Num1 -> Basic Locomotion
Num2 -> Skill 2
Num3 -> Skill 3
Num4 -> Skill 4
Num5 -> Skill 5
Num6 -> Skill 6
Num7 -> Skill 7
Num8 -> Skill 8
```

Do not assume which skill number maps to `gangnam_style` or `dance_102`. Inspect the repo:

```bash
grep -R "gangnam" -n .
grep -R "dance_102" -n .
grep -R "whole_body_tracking" -n src policy include . | head -200
grep -R "Skill" -n src include policy . | head -200
grep -R "Num" -n src include . | head -200
grep -R "FSM\|finite\|state" -n src include . | head -200
```

Find the mapping between:

- keyboard/gamepad skill selection,
- FSM state,
- policy folder,
- config file,
- loaded policy model.

If needed, document the selected config and the exact key sequence required to start the dance.

---

## Step 9 — Run the Dance and Observe

Suggested safe sequence:

```text
1. Launch simulation.
2. Reset if needed.
3. Press Num0 for default pose.
4. Wait until the robot stabilizes.
5. Enable the appropriate skill or dance state.
6. Observe for at least 60 seconds.
7. If robot falls, reset and document the failure point.
```

Record:

- exact launch command,
- selected skill/policy,
- key sequence used,
- whether the robot danced,
- duration before fall or instability,
- terminal logs,
- any MuJoCo warnings,
- frame rate or real-time factor if visible.

If possible, save a short video or screenshots. Do not rely on memory.

---

## Step 10 — Troubleshooting Checklist

### If the robot falls immediately

Check:

```text
1. Did you press Num0 and wait for the default pose before skill switching?
2. Are you using `g1` with the correct scene, likely `scene_29dof`?
3. Is the correct policy config loaded?
4. Does joint order match the model?
5. Are PD gains and action scales correct?
6. Is the policy file present and readable?
7. Is the motion policy intended for this exact G1 morphology?
```

Commands:

```bash
find policy/g1 -type f | sort
grep -R "joint" -n policy/g1 | head -200
grep -R "default_dof_pos\|dof_pos\|kp\|kd\|stiffness\|damping" -n policy/g1 src include | head -240
```

### If no policy appears to load

Check:

```bash
find policy/g1 -type f | sort
grep -R "policy" -n policy/g1 src include | head -200
grep -R "onnx\|libtorch\|\.pt\|\.onnx" -n src include policy/g1 | head -200
```

Confirm whether the build supports `.pt`, `.onnx`, or both.

### If MuJoCo viewer does not open

Check environment and display:

```bash
echo $DISPLAY
echo $WAYLAND_DISPLAY
ldd ./cmake_build/bin/rl_sim_mujoco | grep -i "not found" || true
```

If under WSL2 or headless Linux, document the display error and use Docker/X11 or native Ubuntu.

### If build fails

Record:

```bash
./build.sh -mj 2>&1 | tee build_mujoco.log
```

Then inspect the first actual compiler or linker error. Do not patch randomly.

---

## Optional Docker Route

If local dependencies are difficult, try Docker if the repository supports it.

Expected pattern:

```bash
xhost +local:docker

cd docker
docker compose up -d
docker compose exec rl_sar bash
```

Inside the container:

```bash
./cmake_build/bin/rl_sim_mujoco g1 scene_29dof
```

If Docker instructions differ in the local README, follow the local README.

---

## Deliverables

Create a final report named:

```text
rl_sar_g1_dance_run_report.md
```

The report must include:

1. **Environment**
   - OS and version
   - GPU/CPU if relevant
   - native Linux, WSL2, Docker, or other

2. **Repository state**
   - repository URL
   - commit hash
   - submodule status

3. **Build result**
   - build command used
   - success/failure
   - generated binary path
   - error log summary if failed

4. **Policy discovery**
   - available G1 dance policies
   - exact policy selected
   - `.pt` or `.onnx`
   - relevant config path

5. **Run command**
   - exact command used to launch MuJoCo
   - scene name

6. **Skill activation**
   - key sequence or config change used
   - mapping from skill key to policy if discovered

7. **Observed behavior**
   - whether default pose succeeded
   - whether dance started
   - whether robot stayed upright
   - duration achieved
   - notable failure modes

8. **Artifacts**
   - screenshots or video paths if captured
   - terminal log path
   - build log path if relevant

9. **Next actions**
   - only concrete next steps based on observed failures
   - no invented timeline

---

## Acceptance Criteria

The sprint is successful if all of the following are true:

```text
1. Repository is cloned recursively.
2. MuJoCo target builds or build failure is clearly diagnosed.
3. G1 policy folders are located and documented.
4. G1 MuJoCo simulation launches or launch failure is clearly diagnosed.
5. G1 reaches default pose, or default-pose failure is documented.
6. A dance skill is triggered, or the exact missing mapping/config blocker is documented.
7. A final report `rl_sar_g1_dance_run_report.md` is produced.
```

Stretch goal:

```text
The G1 performs a pretrained dance for at least 60 seconds in MuJoCo without falling.
```

---

## Notes on Custom Kung-Fu Motions

Do not attempt custom kung-fu training in this sprint.

If asked later, use a separate pipeline:

```text
video / AMASS / LAFAN / mocap
  -> SMPL motion extraction
  -> retarget to Unitree G1
  -> motion cleanup and physical feasibility checks
  -> RL motion-imitation training
  -> MuJoCo sim2sim verification
  -> deployment wrapper
```

For custom motion processing and training, investigate PBHC/KungfuBot separately. This `agent.md` is only for running pretrained `rl_sar` G1 dance simulation.

---

## Final Instruction

Work incrementally. First inspect the local repository and README, then run the smallest safe MuJoCo simulation path. Prefer documenting precise blockers over guessing. Never execute real-hardware deployment commands.
